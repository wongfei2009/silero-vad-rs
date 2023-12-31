use ndarray::{Array, CowArray};
use std::path::Path;

use ort::{
    tensor::OrtOwnedTensor, Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult,
    Session, SessionBuilder, Value,
};

#[derive(Debug, PartialEq)]
pub enum VadResult {
    Silence,
    Speaking,
    Start,
    End,
}

/// A struct representing a Voice Activity Detection (VAD) iterator.
pub struct VadIterator {
    // OnnxRuntime resources
    session: Session, // OnnxRuntime session

    // Model config
    window_size_samples: i64, // The size of the window in samples
    threshold: f32,           // The threshold for speech detection
    threshold_margin: f32,    // The margin for the threshold
    min_silence_samples: u32, // The minimum number of samples for silence
    speech_pad_samples: i32,  // The number of samples to pad speech with

    // Model states
    triggerd: bool,          // Whether speech has been detected
    speech_start: u32,       // The sample index of the start of speech
    speech_end: u32,         // The sample index of the end of speech
    temp_end: u32,           // The temporary sample index of the end of speech
    current_sample: u32,     // The current sample index
    speech_probability: f32, // The probability of speech

    // Inputs
    input: Vec<f32>, // The input audio data
    sr: Vec<i64>,    // The sample rate of the audio as a vector
    _h: Vec<f32>,    // The hidden state of the model
    _c: Vec<f32>,    // The cell state of the model

    // Outputs
    ort_outputs: Vec<Value<'static>>, // The outputs of the OnnxRuntime session

    // Result
    pub result: Option<VadResult>, // The result of the VAD
}

impl VadIterator {
    pub fn reset_states(&mut self) {
        // Call reset before each audio start
        self._h.iter_mut().for_each(|x| *x = 0.0);
        self._c.iter_mut().for_each(|x| *x = 0.0);
        self.triggerd = false;
        self.temp_end = 0;
        self.current_sample = 0;
        self.speech_probability = 0.0;
        self.result = None;
    }
    pub fn predict(&mut self, data: &[f32]) {
        // Infer
        // Create ort tensors
        self.input.clear();
        self.input.extend_from_slice(data);
        self.window_size_samples = self.input.len() as i64;

        let input_array = CowArray::from(
            Array::from_shape_vec(
                (1, self.window_size_samples as usize),
                self.input.iter().map(|&x| x as f32).collect(),
            )
            .unwrap(),
        )
        .into_dyn();

        let sr_array = CowArray::from(
            Array::from_shape_vec(1, self.sr.iter().map(|&x| x as i64).collect()).unwrap(),
        )
        .into_dyn();

        let h_array = CowArray::from(
            Array::from_shape_vec((2, 1, 64), self._h.iter().map(|&x| x as f32).collect()).unwrap(),
        )
        .into_dyn();

        let c_array = CowArray::from(
            Array::from_shape_vec((2, 1, 64), self._c.iter().map(|&x| x as f32).collect()).unwrap(),
        )
        .into_dyn();

        // Clear and add inputs
        let ort_inputs = vec![
            Value::from_array(self.session.allocator(), &input_array).unwrap(),
            Value::from_array(self.session.allocator(), &sr_array).unwrap(),
            Value::from_array(self.session.allocator(), &h_array).unwrap(),
            Value::from_array(self.session.allocator(), &c_array).unwrap(),
        ];

        // Infer
        self.ort_outputs = self.session.run(ort_inputs).unwrap();

        // Output probability & update h,c recursively
        let output: OrtOwnedTensor<f32, _> = self.ort_outputs[0].try_extract().unwrap();
        output
            .view()
            .iter()
            .for_each(|&x| self.speech_probability = x);

        let hn = self.ort_outputs[1].try_extract().unwrap();
        self._h.clone_from_slice(hn.view().as_slice().unwrap());
        let cn = self.ort_outputs[2].try_extract().unwrap();
        self._c.clone_from_slice(cn.view().as_slice().unwrap());

        // Push forward sample index
        self.current_sample += self.window_size_samples as u32;

        // Reset temp_end when > threshold
        if self.speech_probability >= self.threshold && self.temp_end != 0 {
            self.temp_end = 0;
        }

        // 1) Silence
        if self.speech_probability < self.threshold && !self.triggerd {
            self.result = Some(VadResult::Silence);
        }
        // 2) Speaking
        if self.speech_probability >= self.threshold - self.threshold_margin && self.triggerd {
            self.result = Some(VadResult::Speaking);
        }
        // 3) Start
        if self.speech_probability >= self.threshold && !self.triggerd {
            self.triggerd = true;
            self.speech_start = self.current_sample
                - self.window_size_samples as u32
                - self.speech_pad_samples as u32; // minus window_size_samples to get precise start time point.
            self.result = Some(VadResult::Start);
        }
        // 4) End
        if self.speech_probability < self.threshold - self.threshold_margin && self.triggerd {
            if self.temp_end == 0 {
                self.temp_end = self.current_sample;
            }
            // a. silence < min_slience_samples, continue speaking
            if self.current_sample - self.temp_end < self.min_silence_samples {
                self.result = Some(VadResult::Speaking);
            }
            // b. silence >= min_slience_samples, end speaking
            else {
                self.speech_end = if self.temp_end != 0 {
                    self.temp_end + self.speech_pad_samples as u32
                } else {
                    self.current_sample + self.speech_pad_samples as u32
                };
                self.temp_end = 0;
                self.triggerd = false;
                self.result = Some(VadResult::End);
            }
        }
    }
    // Construction
    pub fn new(
        model: &Path,
        sample_rate: i32,
        threshold: f32,
        threshold_margin: f32,
        min_silence_duration_ms: i64,
        speech_pad_ms: i64,
    ) -> OrtResult<Self> {
        let environment = Environment::builder()
            .with_execution_providers([
                ExecutionProvider::CUDA(Default::default()),
                ExecutionProvider::CPU(Default::default()),
            ])
            .build()?
            .into_arc();
        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .with_model_from_file(model)?;

        let sr_per_ms = (sample_rate as i64 / 1000) as i32;
        let min_silence_samples = (sr_per_ms as i64 * min_silence_duration_ms) as u32;
        let speech_pad_samples = (sr_per_ms as i64 * speech_pad_ms) as i32;
        let window_size_samples = 256;

        let input = vec![0.0; window_size_samples as usize];
        let sr = vec![sample_rate as i64];
        let _h = vec![0.0; 2 * 1 * 64];
        let _c = vec![0.0; 2 * 1 * 64];

        Ok(Self {
            session,
            window_size_samples,
            threshold,
            threshold_margin,
            min_silence_samples,
            speech_pad_samples,
            triggerd: false,
            speech_start: 0,
            speech_end: 0,
            temp_end: 0,
            current_sample: 0,
            speech_probability: 0.0,
            input,
            sr,
            _h,
            _c,
            ort_outputs: vec![],
            result: None,
        })
    }
}

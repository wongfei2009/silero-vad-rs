pub mod model;

#[cfg(test)]
mod tests {
    use crate::model::{VadIterator, VadResult};
    use std::path::Path;
    #[test]
    fn it_works() {
        let path = Path::new("silero-vad/files/silero_vad.onnx");

        let test_sr: i32 = 8000;
        let test_threshold = 0.5f32;
        let test_threshold_margin = 0.05f32;
        let test_min_silence_duration_ms = 0;
        let test_speech_pad_ms = 0;
        let test_window_samples = 256;

        let mut vad = VadIterator::new(
            &path,
            test_sr,
            test_threshold,
            test_threshold_margin,
            test_min_silence_duration_ms,
            test_speech_pad_ms
        )
        .unwrap();
        let data = vec![0.1; test_window_samples as usize];
        vad.predict(&data);
        assert_eq!(vad.result, Some(VadResult::Silence));
        vad.reset_states();
        assert_eq!(vad.result, None);
    }
}

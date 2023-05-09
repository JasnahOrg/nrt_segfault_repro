// System
use std::fs::File;
use std::io::Read;
use std::time::Duration;

// Local
use crate::bindings::nrt;
use crate::trn::{allocate_tensors, handler_save_outputs, iterate_tensors, load_tensor_values};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum XLAHardware {
    TRN,
}

#[allow(dead_code)]
pub struct XLARunner {
    hardware: XLAHardware,
}

#[derive(Debug, Clone)]
pub enum Output {
    Bool(Vec<bool>),
    Float32(Vec<f32>),
}

#[derive(Debug, Clone)]
pub struct XLARunResults {
    pub output: Vec<Output>,
    /// The debug_ir human-readable reprsentation of the XLA HLO
    pub debug_ir: Option<String>,
    /// This is the graph exececution time without any compilation time, tensor allocation time, or
    /// output copying time.
    pub runtime: Duration,
}

impl XLARunner {
    /// This inits the Neuron NRT library for Trainium if the trn feature is enabled.
    /// Note that NRT should be initialized only once per process. If nrt_close is called,
    /// subsequent calls to nrt_init will fail. This is a known issue with the current version of NRT.
    pub fn new(hardware: XLAHardware) -> Self {
        {
            if hardware == XLAHardware::TRN {
                // Init NRT
                let result = unsafe {
                    nrt::nrt_init(
                        nrt::nrt_framework_type_t_NRT_FRAMEWORK_TYPE_NO_FW,
                        std::ptr::null() as *const i8,
                        std::ptr::null() as *const i8,
                    )
                };
                assert_eq!(result, nrt::NRT_STATUS_NRT_SUCCESS);
            }
        }
        XLARunner { hardware }
    }

    #[allow(unused_variables)]
    /// This compiles the XLA HLO into a NEFF.
    /// See [AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html).
    pub fn run_trn(
        &self,
        neff_path: &str,
        run_name: &str,
        input_names: &[&str],
        inputs: Vec<Vec<f32>>,
        input_shapes: Vec<Vec<u64>>,
    ) -> Result<XLARunResults, String> {
        {
            assert_eq!(input_names.len(), inputs.len());

            // Read NEFF file into a byte vector
            let mut neff_file = File::open(neff_path.clone())
                .unwrap_or_else(|_| panic!("Unable to open NEFF file {}", neff_path));
            let mut neff_data: Vec<u8> = Vec::new();
            neff_file
                .read_to_end(&mut neff_data)
                .expect("Unable to read NEFF file");
            let neff_size = neff_data.len();

            // Load the model
            let mut model: *mut nrt::nrt_model_t = std::ptr::null_mut();
            assert_eq!(model, std::ptr::null_mut());
            assert!(model.is_null());
            // TODO: In production we will need to set the neuron core ids
            // based on model sharding.
            let result = unsafe {
                nrt::nrt_load(
                    neff_data.as_ptr() as *const _,
                    neff_size,
                    0, // neuron core index to start from
                    1, // number of neuron cores to allocate the model to
                    &mut model as *mut *mut nrt::nrt_model_t,
                )
            };
            assert_ne!(model, std::ptr::null_mut());
            assert!(!model.is_null());
            assert_eq!(result, nrt::NRT_STATUS_NRT_SUCCESS);

            // Allocate input and ouptut tensors
            let mut tensor_info_array: *mut nrt::nrt_tensor_info_array_t = std::ptr::null_mut();
            assert_eq!(tensor_info_array, std::ptr::null_mut());
            assert!(tensor_info_array.is_null());
            let result = unsafe {
                nrt::nrt_get_model_tensor_info(
                    model,
                    &mut tensor_info_array as *mut *mut nrt::nrt_tensor_info_array_t,
                )
            };
            assert_eq!(result, nrt::NRT_STATUS_NRT_SUCCESS);
            let tensor_info_array =
                std::ptr::NonNull::new(tensor_info_array).expect("Error: null tensor_info_array");

            let nrt_inputs = allocate_tensors(
                tensor_info_array,
                nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_INPUT,
            );
            let nrt_inputs = nrt_inputs.expect("Error allocating input tensors");

            let outputs = allocate_tensors(
                tensor_info_array,
                nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_OUTPUT,
            );
            let outputs = outputs.expect("Error allocating output tensors");

            // Note that even if input parameters are not initialized, it will
            // still run and it will still produce values.
            if !inputs.is_empty() {
                let result = load_tensor_values(
                    nrt_inputs,
                    tensor_info_array,
                    nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_INPUT,
                    inputs,
                );
                result.expect("Error loading input tensor values");
            }

            // Run it
            let start = std::time::Instant::now();
            let result = unsafe { nrt::nrt_execute(model, nrt_inputs.as_ptr(), outputs.as_ptr()) };
            let runtime = start.elapsed();
            assert_eq!(
                result,
                nrt::NRT_STATUS_NRT_SUCCESS,
                "nrt_execute failed to run model {}",
                run_name
            );

            // TODO: Instead of saving the outputs to file, get them in a Vec<Vec<f32>>
            // Saving outputs to files
            let result = unsafe {
                iterate_tensors(
                    outputs.as_ptr(),
                    tensor_info_array.as_ptr(),
                    nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_OUTPUT,
                    handler_save_outputs,
                    std::ptr::null_mut(),
                )
            };
            let result = result.expect("Error saving output tensors");
            assert_eq!(result.0, nrt::NRT_STATUS_NRT_SUCCESS);
            let output = result.1;

            unsafe {
                nrt::nrt_destroy_tensor_set(&mut nrt_inputs.as_ptr());
                nrt::nrt_destroy_tensor_set(&mut outputs.as_ptr());
                nrt::nrt_free_model_tensor_info(tensor_info_array.as_ptr());
            };
            //let output = Vec::new();
            return Ok(XLARunResults {
                output,
                debug_ir: None,
                runtime,
            });
        }
        // This will be seen as unreachable code when --feature trn is enabled
        #[allow(unreachable_code)]
        Err("TRN feature is not enabled.".to_string())
    }
}

impl Drop for XLARunner {
    fn drop(&mut self) {
        if self.hardware == XLAHardware::TRN {
            unsafe {
                nrt::nrt_close();
            }
        }
    }
}

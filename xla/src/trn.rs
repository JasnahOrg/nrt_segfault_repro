//! These are functions for running Trainium hardware from Rust.
//! This is based on the NRT API C Code examples [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-api-guide.html#the-code).

// System
use std::ffi::{CStr, OsStr};
use std::fs::OpenOptions;
use std::io::Write;
use std::os::raw::c_void;
use std::os::unix::ffi::OsStrExt;
use std::path::PathBuf;
use std::ptr::NonNull;

// Local
use crate::bindings::nrt;
use crate::xla_runner::Output;

/// A function that can be passed to iterate_tensors
/// to run it once on each tensor.
pub type TensorHandler = unsafe extern "C" fn(
    tensor: *mut nrt::nrt_tensor_t,
    tensor_info: *const nrt::nrt_tensor_info_t,
    result: *mut nrt::NRT_STATUS,
    return_value: &mut Output,
    args: *mut std::ffi::c_void,
) -> bool;

/// A wrapper that iterates tensors and calls the given handler on
/// each tensor.
///
/// # Safety
///
/// This function is marked as unsafe due to the use of raw pointers.
/// To call this function safely, the following invariants must be upheld:
///
/// 1. `tset` must be a valid, non-null pointer to an `nrt_tensor_set_t` instance.
/// 2. `info_array` must be a valid, non-null pointer to an `nrt_tensor_info_array_t` instance,
///    and the instance should be initialized properly.
/// 3. `args` can be null or non-null, depending on the requirements of the provided `handler`.
///
/// This is based on the code [here](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/neuron-runtime/nrt-api-guide.html#the-code>).
///
/// If these invariants are not upheld, the function may cause undefined behavior or memory corruption.
/// TODO: This only supports returning f32, but it will need to support other types.
pub unsafe fn iterate_tensors(
    tset: *mut nrt::nrt_tensor_set_t,
    info_array: *mut nrt::nrt_tensor_info_array_t,
    usage_type: nrt::nrt_tensor_usage_t,
    handler: TensorHandler,
    args: *mut std::ffi::c_void,
) -> Result<(nrt::NRT_STATUS, Vec<Output>), nrt::NRT_STATUS> {
    // Check if tset is a non-null pointer
    if tset.is_null() {
        eprintln!("Invalid tset pointer");
        return Err(nrt::NRT_STATUS_NRT_FAILURE);
    }
    // Check if info_array is a non-null pointer
    if info_array.is_null() {
        eprintln!("Invalid info_array pointer");
        return Err(nrt::NRT_STATUS_NRT_FAILURE);
    }

    let mut final_result = nrt::NRT_STATUS_NRT_SUCCESS;
    let tensor_count = unsafe { (*info_array).tensor_count } as usize;
    let tensor_info_array = unsafe { (*info_array).tensor_array.as_ptr() };

    let mut return_values: Vec<Output> = Vec::new();
    for tensor_idx in 0..tensor_count {
        let tensor_info = unsafe { &*tensor_info_array.add(tensor_idx) };

        if tensor_info.usage != usage_type {
            continue;
        }

        let mut tensor: *mut nrt::nrt_tensor_t = std::ptr::null_mut();
        let result = unsafe {
            nrt::nrt_get_tensor_from_tensor_set(tset, tensor_info.name.as_ptr(), &mut tensor)
        };

        if result != nrt::NRT_STATUS_NRT_SUCCESS {
            continue;
        }

        let mut handler_result = nrt::NRT_STATUS_NRT_SUCCESS;
        let mut return_value = match tensor_info.dtype {
            nrt::nrt_dtype_NRT_DTYPE_FLOAT32 => Output::Float32(Vec::with_capacity(
                tensor_info.size / std::mem::size_of::<f32>(),
            )),
            nrt::nrt_dtype_NRT_DTYPE_UINT8 => Output::Bool(Vec::with_capacity(tensor_info.size)),
            _ => panic!("Unsupported dtype {:?}", tensor_info.dtype),
        };
        //let mut return_value = Vec::with_capacity(tensor_info.size / std::mem::size_of::<f32>());
        if !unsafe {
            handler(
                tensor,
                tensor_info as *const _,
                &mut handler_result,
                &mut return_value,
                args,
            )
        } {
            return Err(handler_result);
        }
        match return_value {
            Output::Float32(ref v) => {
                if !v.is_empty() {
                    return_values.push(Output::Float32(v.clone()));
                }
            }
            Output::Bool(ref v) => {
                if !v.is_empty() {
                    return_values.push(Output::Bool(v.clone()));
                }
            }
        }

        if final_result == nrt::NRT_STATUS_NRT_SUCCESS && handler_result != final_result {
            final_result = handler_result;
        }
    }

    Ok((final_result, return_values))
}

/// Save all the output tensors to file
///
/// # Safety
///
/// This function is marked as unsafe due to the use of raw pointers.
/// To call this function safely, the following invariants must be upheld:
///
/// 1. `tensor` must be a valid, non-null pointer to an `nrt_tensor_t` instance.
/// 2. `tensor_info` must be a valid, non-null pointer to an `nrt_tensor_info_t` instance,
///    and the instance should be initialized properly.
/// 3. `result` must be a valid, non-null pointer to an `nrt::NRT_STATUS` instance.
/// 4. `_args` is currently unused, so it can be null or non-null.
///
/// If these invariants are not upheld, the function may cause undefined behavior or memory corruption.
pub unsafe extern "C" fn handler_save_outputs(
    tensor: *mut nrt::nrt_tensor_t,
    tensor_info: *const nrt::nrt_tensor_info_t,
    result: *mut nrt::NRT_STATUS,
    return_value: &mut Output,
    _args: *mut c_void,
) -> bool {
    // Check if tensor is a non-null pointer
    if tensor.is_null() {
        eprintln!("Invalid tensor pointer");
        return false;
    }

    // Check if tensor_info is a non-null pointer
    if tensor_info.is_null() {
        eprintln!("Invalid tensor_info pointer");
        return false;
    }

    // Check if result is a non-null pointer
    if result.is_null() {
        eprintln!("Invalid result pointer");
        return false;
    }

    let tensor_info_name = CStr::from_ptr((*tensor_info).name.as_ptr())
        .to_str()
        .unwrap();
    let tensor_data =
        std::alloc::alloc(std::alloc::Layout::from_size_align((*tensor_info).size, 1).unwrap())
            as *mut c_void;

    if tensor_data.is_null() {
        eprintln!(
            "Unable to allocate memory for saving output tensor {}",
            tensor_info_name
        );
        *result = nrt::NRT_STATUS_NRT_FAILURE;
        return true;
    }

    *result = nrt::nrt_tensor_read(tensor, tensor_data, 0, (*tensor_info).size);
    if *result != nrt::NRT_STATUS_NRT_SUCCESS {
        eprintln!("Unable to read tensor {}", tensor_info_name);
        std::alloc::dealloc(
            tensor_data as *mut u8,
            std::alloc::Layout::from_size_align((*tensor_info).size, 1).unwrap(),
        );
        return true;
    }

    let mut filename = PathBuf::from(<OsStr as OsStrExt>::from_bytes(tensor_info_name.as_bytes()));
    filename.set_extension("out");

    let mut file = match OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&filename)
    {
        Ok(file) => file,
        Err(_) => {
            eprintln!("Unable to open {} for writing", filename.to_string_lossy());
            std::alloc::dealloc(
                tensor_data as *mut u8,
                std::alloc::Layout::from_size_align((*tensor_info).size, 1).unwrap(),
            );
            *result = nrt::NRT_STATUS_NRT_FAILURE;
            return true;
        }
    };

    let write_result = file.write_all(std::slice::from_raw_parts(
        tensor_data as *const u8,
        (*tensor_info).size,
    ));

    match write_result {
        Ok(_) => {
            //println!(
            //"Wrote tensor {} contents to file {}",
            //tensor_info_name,
            //filename.to_string_lossy()
            //);
        }
        Err(_) => {
            eprintln!(
                "Unable to write tensor {} contents to file {}",
                tensor_info_name,
                filename.to_string_lossy()
            );
            *result = nrt::NRT_STATUS_NRT_FAILURE;
        }
    }
    //for i in 0..((*tensor_info).size / std::mem::size_of::<f32>()) {
    //let value = *(tensor_data as *const f32).add(i);
    //return_value.push(value);
    //}
    match *return_value {
        Output::Float32(ref mut v) => {
            if (*tensor_info).dtype == nrt::nrt_dtype_NRT_DTYPE_FLOAT32 {
                for i in 0..((*tensor_info).size / std::mem::size_of::<f32>()) {
                    let value = *(tensor_data as *const f32).add(i);
                    v.push(value);
                }
            } else {
                eprintln!(
                    "Mismatched data type. Expected f32 but got {:?}",
                    (*tensor_info).dtype
                );
                *result = nrt::NRT_STATUS_NRT_FAILURE;
                return true;
            }
        }
        Output::Bool(ref mut v) => {
            if (*tensor_info).dtype == nrt::nrt_dtype_NRT_DTYPE_UINT8 {
                for i in 0..(*tensor_info).size {
                    let value = (*(tensor_data as *const u8).add(i)) != 0;
                    v.push(value);
                }
            } else {
                eprintln!(
                    "Mismatched data type. Expected bool but got {:?}",
                    (*tensor_info).dtype
                );
                *result = nrt::NRT_STATUS_NRT_FAILURE;
                return true;
            }
        }
    }

    std::alloc::dealloc(
        tensor_data as *mut u8,
        std::alloc::Layout::from_size_align((*tensor_info).size, 1).unwrap(),
    );

    true
}

/// This is used to load the given values into the input tensors
/// of the given tensor set.
///
/// # Safety
///
/// This function is marked as unsafe due to the use of raw pointers.
/// To call this function safely, the following invariants must be upheld:
///
/// 1. `tensors` must be a valid, non-null pointer to an `nrt_tensor_set_t` instance.
/// 2. `info_array` must be a valid, non-null pointer to an `nrt_tensor_info_array_t` instance,
///    and the instance should be initialized properly.
/// 3. The `tensor_count` and `tensor_array` fields of `info_array` must be correctly initialized.
/// 4. The `tensor_array` field must point to an array with at least `tensor_count` elements,
///    and each element must be a properly initialized `nrt_tensor_info`.
/// 5. The `usage_type` parameter must be a valid `nrt_tensor_usage_t` value.
/// 6. The `values` Vec should have a length equal to `tensor_count`, and each inner Vec should
///    have a length that matches the size of the corresponding tensor.
///
/// If these invariants are not upheld, the function may cause undefined behavior or memory corruption.
/// TODO: Generalize this to non-f32 types
pub fn load_tensor_values(
    tensors: NonNull<nrt::nrt_tensor_set_t>,
    info_array: NonNull<nrt::nrt_tensor_info_array_t>,
    usage_type: nrt::nrt_tensor_usage_t,
    values: Vec<Vec<f32>>,
) -> Result<(), nrt::NRT_STATUS> {
    if values.is_empty() {
        return Ok(());
    }

    // Check that usage_type is valid
    if usage_type != nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_INPUT
        && usage_type != nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_OUTPUT
    {
        return Err(nrt::NRT_STATUS_NRT_INVALID);
    }

    // Retrieve tensor_count and tensor_info_array safely
    let tensor_count = unsafe { info_array.as_ref().tensor_count as usize };
    let tensor_info_array = unsafe { info_array.as_ref().tensor_array.as_ptr() };

    // Validate the tensor_count and tensor_info_array
    if tensor_count == 0 {
        return Err(nrt::NRT_STATUS_NRT_INVALID);
    }

    let mut num_tensors_loaded = 0;

    for (tensor_idx, data) in values.iter().enumerate() {
        let tensor_info = unsafe { &*tensor_info_array.add(tensor_idx) };

        let expected_data_length = tensor_info.size / std::mem::size_of::<f32>();
        if data.len() != expected_data_length {
            return Err(nrt::NRT_STATUS_NRT_INVALID);
        }

        if tensor_info.usage != usage_type {
            continue;
        }

        let mut tensor: *mut nrt::nrt_tensor_t = std::ptr::null_mut();
        let result = unsafe {
            nrt::nrt_get_tensor_from_tensor_set(
                tensors.as_ptr(),
                tensor_info.name.as_ptr(),
                &mut tensor as *mut *mut nrt::nrt_tensor_t,
            )
        };
        if result != nrt::NRT_STATUS_NRT_SUCCESS {
            return Err(result);
        }

        // Get a pointer to the first element in the Vec
        let data_ptr = data.as_ptr();
        // Cast the pointer to a *const c_void
        let data_c_void_ptr = data_ptr as *const c_void;
        let tensor_size = data.len() * std::mem::size_of::<f32>();

        let result = unsafe { nrt::nrt_tensor_write(tensor, data_c_void_ptr, 0, tensor_size) };
        if result != nrt::NRT_STATUS_NRT_SUCCESS {
            return Err(result);
        }
        num_tensors_loaded += 1;
    }
    if num_tensors_loaded != values.len() {
        let len = values.len();
        eprintln!(
            "The number of tensors in the model {tensor_count} does not match the number of values provided {len}"
        );
        return Err(nrt::NRT_STATUS_NRT_FAILURE);
    }
    Ok(())
}

/// Initializes tensor memory in the Trainium hardware.
///
/// # Safety
///
/// This function is marked as unsafe due to the use of raw pointers.
/// To call this function safely, the following invariants must be upheld:
///
/// 1. `info_array` must be a valid, non-null pointer to an `nrt_tensor_info_array_t` instance,
///    and the instance should be initialized properly.
/// 2. The `tensor_count` and `tensor_array` fields of `info_array` must be correctly initialized.
/// 3. The `tensor_array` field must point to an array with at least `tensor_count` elements,
///    and each element must be a properly initialized `nrt_tensor_info`.
/// 4. The `usage_type` parameter must be a valid `nrt_tensor_usage_t` value.
///
/// If these invariants are not upheld, the function may cause undefined behavior or memory corruption.
pub fn allocate_tensors(
    info_array: NonNull<nrt::nrt_tensor_info_array_t>,
    usage_type: nrt::nrt_tensor_usage_t,
) -> Result<NonNull<nrt::nrt_tensor_set_t>, nrt::NRT_STATUS> {
    // Check that usage_type is valid
    if usage_type != nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_INPUT
        && usage_type != nrt::nrt_tensor_usage_NRT_TENSOR_USAGE_OUTPUT
    {
        return Err(nrt::NRT_STATUS_NRT_INVALID);
    }

    let mut out_tset: *mut nrt::nrt_tensor_set_t = std::ptr::null_mut();
    let result =
        unsafe { nrt::nrt_allocate_tensor_set(&mut out_tset as *mut *mut nrt::nrt_tensor_set_t) };
    if result != nrt::NRT_STATUS_NRT_SUCCESS {
        return Err(result);
    }

    let out_tset = NonNull::new(out_tset).ok_or(nrt::NRT_STATUS_NRT_INVALID)?;

    // Retrieve tensor_count and tensor_info_array safely
    let tensor_count = unsafe { info_array.as_ref().tensor_count as usize };
    let tensor_info_array = unsafe { info_array.as_ref().tensor_array.as_ptr() };

    // Validate the tensor_count
    if tensor_count == 0 {
        return Err(nrt::NRT_STATUS_NRT_INVALID);
    }

    for tensor_idx in 0..tensor_count {
        let tensor_info = unsafe { &*tensor_info_array.add(tensor_idx) };

        if tensor_info.usage != usage_type {
            continue;
        }

        let mut tensor: *mut nrt::nrt_tensor_t = std::ptr::null_mut();
        let result = unsafe {
            nrt::nrt_tensor_allocate(
                nrt::nrt_tensor_placement_t_NRT_TENSOR_PLACEMENT_DEVICE,
                0,
                tensor_info.size,
                tensor_info.name.as_ptr(),
                &mut tensor as *mut *mut nrt::nrt_tensor_t,
            )
        };

        if result != nrt::NRT_STATUS_NRT_SUCCESS {
            return Err(result);
        }

        let tensor = NonNull::new(tensor).ok_or(nrt::NRT_STATUS_NRT_INVALID)?;

        let result = unsafe {
            nrt::nrt_add_tensor_to_tensor_set(
                out_tset.as_ptr(),
                tensor_info.name.as_ptr(),
                tensor.as_ptr(),
            )
        };

        if result != nrt::NRT_STATUS_NRT_SUCCESS {
            return Err(result);
        }
    }

    Ok(out_tset)
}

#[cfg(test)]
mod tests {
    // Local
    use crate::xla_runner::{XLAHardware, XLARunner};

    #[test]
    fn transformer_xla_benchmark() {
        let runner = XLARunner::new(XLAHardware::TRN);
        let run_name = "transformer_test";

        #[allow(unused_variables)]
        let segfault_path = "./transformer_xla_segfault.neff";
        #[allow(unused_variables)]
        let working_path = "./transformer_xla_working.neff";
        let inputs = vec![];
        let input_names = vec![];
        let input_shapes = vec![];
        assert_eq!(input_names.len(), inputs.len());
        assert_eq!(input_shapes.len(), inputs.len());

        runner
            .run_trn(
                &segfault_path,
                //&working_path,
                run_name,
                &input_names,
                inputs,
                input_shapes,
            )
            .unwrap();
        println!("Done");
    }
}

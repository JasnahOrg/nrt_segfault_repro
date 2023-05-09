#![allow(non_upper_case_globals)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(deref_nullptr)]
#![allow(unused)]

#[allow(clippy::all)]
pub mod nrt {
    #![allow(rustdoc::broken_intra_doc_links)]
    include!(concat!(env!("OUT_DIR"), "/nrt_bindings.rs"));
}

#[cfg(test)]
mod tests {
    #[test]
    fn nrt_test_bindings_loaded() {
        unsafe {
            super::nrt::nrt_init(
                super::nrt::nrt_framework_type_t_NRT_FRAMEWORK_TYPE_NO_FW,
                std::ptr::null() as *const i8,
                std::ptr::null() as *const i8,
            );
        }
    }
}

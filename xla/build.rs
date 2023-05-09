// Std
use std::env;
use std::io::Result;
use std::path::PathBuf;

// Third Party
extern crate bindgen;

fn generate_bindings(name: &str) {
    println!("Called generate_bindings!");
    // Tell cargo to invalidate the built crate whenever the wrapper changes
    println!("cargo:rerun-if-changed={}_wrapper.h", name);

    // Tell cargo to tell rustc to link required system shared libraries
    println!("cargo:rustc-link-lib=dylib={}", name);

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(format!("{}_wrapper.h", name))
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    // Write the bindings to the $OUT_DIR/bindings.rs file.
    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join(format!("{}_bindings.rs", name)))
        .expect("Couldn't write bindings!");
}

fn main() -> Result<()> {
    generate_bindings("nrt");

    Ok(())
}

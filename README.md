### Setup
- Create a trn1.2xlarge instance with with `Deep Learning AMI Neuron PyTorch 1.13 (Ubuntu 20.04) 20230505`.
- Install Rust: `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- Make rust available in the current shell: `source "$HOME/.cargo/env"`
- Install clang
    - `sudo apt-get update`
    - `sudo apt-get install -y libclang-dev`
- Make NRT findable at compile time:
```
export CPATH=/opt/aws/neuron/include:$CPATH
export LIBRARY_PATH=/opt/aws/neuron/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/opt/aws/neuron/lib:$LD_LIBRARY_PATH
```

### Reproduce Segfault
- Run the test: `cargo test transformer_xla_benchmark -- --show-output --nocapture`
- Expected output:
```
   Compiling xla v0.0.1 (/home/ubuntu/dev/nrt_segfault_repro/xla)
    Finished test [unoptimized + debuginfo] target(s) in 0.29s
     Running unittests src/lib.rs (target/debug/deps/xla-181b9a97af6e1b2a)

running 1 test
error: test failed, to rerun pass `--lib`

Caused by:
  process didn't exit successfully: `/home/ubuntu/dev/nrt_segfault_repro/target/debug/deps/xla-181b9a97af6e1b2a transformer_xla_benchmark --show-output --nocapture` (signal: 11, SIGSEGV: invalid memory reference)
```

### Overview
- This contains two .neffs:
    - transformer_xla_working.neff, this is a Transformer with (n_context 6, n_layers 1, d_model 4096, n_heads 8)
    - transformer_xla_segfault.neff, this is a Transformer with (n_context 2048, n_layers 1, d_model 5120, n_heads 80)
    - The neffs have no parameter inputs. The input to the Transformer is generated from a uniform distribution as part of the XLA graph.
- The test being run can be found at the bottom of xla/src/trn.rs. Switch the path on the `run_trn()` call to run the working .neff rather than the segfault neff. Expected output from the working .neff:
```
   Compiling xla v0.0.1 (/home/ubuntu/dev/nrt_segfault_repro/xla)
    Finished test [unoptimized + debuginfo] target(s) in 0.28s
     Running unittests src/lib.rs (target/debug/deps/xla-181b9a97af6e1b2a)

running 1 test
Done
test trn::tests::transformer_xla_benchmark ... ok

successes:

successes:
    trn::tests::transformer_xla_benchmark

test result: ok. 1 passed; 0 failed; 0 ignored; 0 measured; 8 filtered out; finished in 9.49s
```
- The primary function of interest `run_trn` in xla_runner.rs, which calls helper functions in trn.rs.

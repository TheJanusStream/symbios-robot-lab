# Symbios Robot Lab

An evolutionary robotics sandbox that co-evolves robot **bodies** (L-System morphologies) and **brains** (NEAT or HyperNEAT neural networks) in a 3D physics simulation.

Built with [Bevy](https://bevyengine.org/) and [Avian3D](https://github.com/Jondolf/avian) physics, on the Symbios crates.

## Features

- **Co-evolution of body and brain**: L-System grammars define the robot's structure (parts, joints, sensors); a neural network drives the joints. Both are crossed over and mutated together.
- **Two brain encodings**: HyperNEAT (the default) evolves a pattern, a CPPN, that paints the network onto the body, so one genome drives any body; direct NEAT evolves the network itself. See [Brains](#brains).
- **Five evolutionary algorithms**: NSGA-II (multi-objective), SimpleGA (elitist tournament), grid MAP-Elites and CVT-MAP-Elites (quality-diversity archives), and novelty search.
- **Sensors**: IMU (accelerations), touch, ultrasonic range, hinge angles and slider extensions, alongside a 1 Hz clock (its sine and cosine). Each driven joint's output is a target position within its own limits, reached by a motor held to the effort its limit gives (an evolvable constant, capped).
- **Joints**: driven hinges and sliders, passive ball joints with twist and swing limits, and welds.
- **Anti-"vibrobot" fitness**: forward displacement times seven quality factors (posture, smoothness, control jerk, body jerk, a minimum speed, foot step cycles, centre-of-mass bounce), so robots that slide by vibrating score near zero.
- **Live UI**: an egui panel with algorithm selection, parameter sliders, fitness graphs, the Pareto front and the MAP-Elites archives. Click a robot to inspect its live brain inputs.
- **Headless runs**: the evolution loop runs without a window, for tests, batch runs and experiments. A seed repeats a whole run exactly.
- **WebAssembly**: deployed to GitHub Pages.

## Getting started

### Native

```bash
cargo run
```

Evolution starts as soon as the window opens; **Restart evolution** starts over. Drop `.lsys` files into `assets/morphologies/` and they join the pool of fresh bodies (native only).

### Headless

```bash
cargo run --release --example evolve -- --generations 30 --seed 1
```

One row per generation (best and mean fitness, the parent pool's best, the furthest robot, explosions). Options: `--population`, `--duration`, `--algorithm simple|nsga2|map-elites|cvt|novelty`, `--brain hyperneat|direct`, `--mutation-rate`, `--csv PATH`. From code, `symbios_robot_lab::headless::HeadlessLab` runs the same loop.

### Web (WASM)

```bash
cargo install wasm-bindgen-cli --version 0.2.128  # must match the wasm-bindgen pin in Cargo.toml
cargo build --profile wasm-release --target wasm32-unknown-unknown --bin symbios-robot-lab
wasm-bindgen --out-dir ./dist --target web \
  --out-name symbios-robot-lab --no-typescript \
  target/wasm32-unknown-unknown/wasm-release/symbios-robot-lab.wasm
cp index.html ./dist/
```

Serve `./dist/` with any static file server. The deploy also runs `wasm-opt -Oz`.

### Checks

CI runs the same commands:

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo clippy --target wasm32-unknown-unknown -- -D warnings
cargo test
```

## Architecture

The library holds everything that does not draw; the binary adds the window.

```text
src/
  lib.rs, headless.rs   the library and its headless runner
  core/                 config, genotypes, population, presets, hot-loaded morphologies
  simulation/           SimulationPlugin: bodies, sensors, brains, metrics, fitness,
                        the parent pool and the generational lifecycle, on fixed steps
  main.rs               the windowed app
  ui/                   egui control panel and robot inspector
  visuals/              scene, robot meshes and materials, camera follow
examples/evolve.rs      headless runs from the command line
tests/
  headless_evolution.rs whole generations without a window
  evolution_pipeline.rs NEAT mutation, crossover, selection pressure
  morphology_pipeline.rs L-System to blueprint topology
```

### Simulation loop

Every generation lasts a fixed number of physics steps (default 15 simulated seconds at 64 Hz; the window shows them at 5x):

1. **Spawn**: each agent's L-System genotype is derived into a body, and its brain is compiled into a feed-forward network for that body (see [Brains](#brains)).
2. **Evaluate**: every fixed step, sensors fill the brain's inputs, the network drives the joint motors, and metrics accumulate. The first third of the generation is settling and is not measured.
3. **Score**: the walker fitness above; NSGA-II adds the inverse Cost of Transport as a second objective. Robots that explode score below every intact one.
4. **Breed**: parents are selected by the active algorithm, crossed over and mutated; a tenth of each generation is fresh bodies from the morphology presets.

### Brains

Both encodings evolve a NEAT genome and run the same kind of feed-forward network, with one input per sensor value and one `tanh` output per driven joint. An output picks the joint's target position within its range, and the joint's motor servoes there.

- **Direct NEAT**: the genome is the network. Inputs and outputs are numbered, so when crossover or mutation changes the body, the network is resized and the numbers shift meaning.
- **HyperNEAT** (`src/simulation/hyperneat.rs`): the genome is a CPPN, and the network is painted from it for each body, through symbios-neat's `SubstratePainter`. Every input except the clock sits at a point on the body: an IMU's channels around the IMU, a touch or distance sensor at its mount, a hinge's angle at the hinge, a slider's extension at the slider. Each driven joint has an interneuron and a motor neuron where it sits. The CPPN is asked, for each pair of neurons, given both positions and their distance, whether they link and how strongly; and for each joint, its biases and the weights from the clock's sine and cosine (its rhythm's phase and strength). Each kind of link has its own CPPN output, and a fresh CPPN links only neurons near each other, a rule evolution can reshape. Because the brain depends on where things are rather than how they are numbered, an offspring that grows a leg gets a controller for it, and legs placed alike move alike.

Measured with the headless runner (16 to 32 seeds per setting, 30 generations, NSGA-II, 64 robots, 15 s), HyperNEAT reaches a final best fitness of about 1.8 (1.79 to 1.88 with weight limits from 0.25 to 0.6; the lab uses 0.5) against 1.48 for direct NEAT. Two choices made the difference. A fresh CPPN links only neurons near each other: with every link expressed, 84% of fresh motors sat pinned at a limit. And painted weights stay small, so motor neurons follow the clock as smooth waves instead of slamming between their limits: with weights up to 3, HyperNEAT reached only 0.63.

### Morphology presets

Generation 0 draws from eight legged walkers (`src/core/presets.rs`): a millipede, a quadruped, a hexapod, a three-pair insect and a low crawler, plus three that bring in the other joints and sensors: the Tripod, whose shins slide; the Strider, on flat feet with ball-joint ankles; and the Scout, a crawler with an ultrasonic eye watching the ground ahead. Evolution modifies them through parametric jitter, structural insertion and deletion, operator flips, rule duplication and topological swaps.

## Dependencies

| Crate | Purpose |
| --- | --- |
| `bevy` | ECS engine and renderer (only the features the lab uses) |
| `avian3d` | 3D rigid-body physics |
| `bevy_egui` | Immediate-mode UI |
| `bevy_panorbit_camera` | Orbit camera controls |
| `symbios` | L-System parsing, derivation, mutation, crossover |
| `symbios-robot` | L-System to robot blueprint interpreter |
| `symbios-neat` | NEAT genomes, CPPNs, substrate painting and networks |
| `symbios-genetics` | Genotype and phenotype traits, NSGA-II, CVT, novelty |
| `bevy_symbios` | The `.lsys` asset loader for hot-loading (native only) |

## License

See [LICENSE](LICENSE).

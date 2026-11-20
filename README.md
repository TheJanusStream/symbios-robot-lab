# Symbios Robot Lab

An evolutionary robotics sandbox that co-evolves robot **morphology** (L-System bodies) and **brains** (NEAT neural networks) in a 3D physics simulation.

Built with [Bevy](https://bevyengine.org/) and [Avian3D](https://github.com/Jondolf/avian) physics.

## Features

- **Co-evolution of body and brain** — L-System grammars define the robot's physical structure (modules, joints, sensors), while a NEAT network controls the joints. Both are mutated and crossed over together.
- **Three evolutionary algorithms** — NSGA-II (multi-objective), SimpleGA (single-objective with elitism), and MAP-Elites (quality-diversity archive).
- **Rich sensor model** — IMU (6-axis acceleration), touch (contact boolean), and joint proprioception (hinge angle) feed into the neural controller alongside oscillation clocks.
- **Real-time UI** — Egui control panel with algorithm selection, parameter sliders, fitness graphs, Pareto front scatter plot, and MAP-Elites archive heatmap. Click any robot to inspect its live brain inputs and topology.
- **WebAssembly support** — Deploys to GitHub Pages via a CI workflow.

## Getting started

### Native

```bash
cargo run
```

Click **Initialize Population** in the UI panel to start evolution.

### Web (WASM)

```bash
cargo install wasm-bindgen-cli --version 0.2.128  # must match the wasm-bindgen pin in Cargo.toml
cargo build --release --target wasm32-unknown-unknown
wasm-bindgen --out-dir ./dist --target web \
  --out-name symbios-robot-lab --no-typescript \
  target/wasm32-unknown-unknown/release/symbios-robot-lab.wasm
cp index.html ./dist/
```

Serve `./dist/` with any static file server.

### Tests

```bash
cargo test
```

## Architecture

```text
src/
  core/           Config, genotypes, population state, morphology presets
  simulation/     Arena, physics, neural evaluation, scoring, generational lifecycle
  ui/             Egui lab panel and robot inspector
  visuals/        Robot spawning/rendering, materials, camera control
tests/
  evolution_pipeline.rs     NEAT mutation, crossover, selection pressure
  morphology_pipeline.rs    L-System → RobotBlueprint topology validation
```

### Simulation loop

Each generation runs for a configurable duration (default 15 s at 5x time scale):

1. **Spawn** — Robots are placed on a grid. Each body is derived from its L-System genotype; the NEAT brain is compiled into a feed-forward evaluator.
2. **Evaluate** — Every physics tick: gather sensor inputs, run the neural network, drive joint motors. A settling period (first third) is excluded from fitness measurement.
3. **Score** — Fitness = forward displacement x average up-alignment x (average height + 0.5). NSGA-II adds a second objective for energy efficiency.
4. **Select & breed** — Parents are selected via the active algorithm; offspring are produced by L-System crossover + NEAT crossover, then mutated. 10% of each generation is injected fresh from the morphology preset pool.

### Morphology presets

Generation 0 draws from a curated pool of L-System body plans: millipede, quadruped, hexapod, sidewinder snake, biped, and starfish. Evolution can modify these through parametric jitter, structural insertion/deletion, operator flips, rule duplication, and topological swaps.

## Dependencies

| Crate | Purpose |
| --- | --- |
| `bevy` | ECS game engine and renderer |
| `avian3d` | 3D rigid-body physics |
| `bevy_egui` | Immediate-mode UI |
| `bevy_panorbit_camera` | Orbit camera controls |
| `symbios` | L-System parsing, derivation, mutation, crossover |
| `symbios-robot` | L-System to RobotBlueprint interpreter |
| `symbios-neat` | NEAT genome, CPPN evaluator |
| `symbios-genetics` | Genotype/Phenotype traits, NSGA-II |
| `bevy_symbios` | Bevy integration (materials, robot spawning) |

## License

See [LICENSE](LICENSE).

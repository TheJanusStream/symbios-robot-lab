# Symbios Robot Lab

An evolutionary robotics sandbox that co-evolves robot **bodies** (L-System morphologies) and **brains** (NEAT or HyperNEAT neural networks) in a 3D physics simulation.

Built with [Bevy](https://bevyengine.org/) and [Avian3D](https://github.com/Jondolf/avian) physics, on the Symbios crates.

## Features

- **Co-evolution of body and brain**: L-System grammars define the robot's structure (parts, joints, sensors); a neural network drives the joints. Each child varies its body or its brain, and new bodies can be protected while their brains catch up. See [Breeding](#breeding).
- **Two brain encodings**: HyperNEAT (the default) evolves a pattern, a CPPN, that paints the network onto the body, so one genome drives any body; direct NEAT evolves the network itself. See [Brains](#brains).
- **Six evolutionary algorithms**: NSGA-II (multi-objective), SimpleGA (elitist tournament), grid MAP-Elites and CVT-MAP-Elites (quality-diversity archives), novelty search, and Dominated Novelty Search (quality diversity without a grid). The quality-diversity algorithms spread robots over their gait, their body, or both. See [Quality diversity](#quality-diversity).
- **Sensors**: IMU (accelerations), touch, ultrasonic range, hinge angles and slider extensions, alongside a clock (its sine and cosine) whose frequency, 0.25 to 4 Hz, each robot evolves. Each driven joint's output is a target position within its own limits, reached by a motor held to the effort its limit gives (an evolvable constant, capped).
- **Joints**: driven hinges and sliders, passive ball joints with twist and swing limits, and welds.
- **Terrain**: the flat floor, or rough (bumps up to 20 cm) or hilly (hills up to 3 m) ground from [bevy_symbios_ground](https://github.com/TheJanusStream/bevy_symbios_ground), laid afresh each generation from the run's seed and the generation number, so no gait wins by fitting one patch of ground and runs with the same seed walk the same grounds. Heights and bounce are measured above the ground under the body.
- **Two tasks**: walk forward, or walk to a target whose bearing the brain reads (placed by the user in the window, or drawn anew each generation). See [Walking to a target](#walking-to-a-target).
- **Anti-"vibrobot" fitness**: forward displacement (or progress toward the target) times seven quality factors (posture, smoothness, control jerk, body jerk, a minimum speed, foot step cycles, centre-of-mass bounce), so robots that slide by vibrating score near zero.
- **Live UI**: an egui panel with algorithm selection, parameter sliders, fitness graphs, the Pareto front, the quality-diversity archives and the parents' spread. Click a robot to inspect its body, brain and live inputs.
- **Break a leg**: lame any robot from its inspector (its leg's joints go limp) to see how its gait copes, or evolve with every robot missing a random leg. See [Leg damage](#leg-damage).
- **Hall of fame**: the session's best robots, to watch walking on their own, copy or save as JSON, import (pasted, or dropped onto the window on the desktop), or send back into evolution. See [Hall of fame](#hall-of-fame).
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

One row per generation (best and mean fitness, the parent pool's best, the furthest robot, explosions, the pool's distinct bodies and its oldest body's age). Options: `--population`, `--duration`, `--algorithm simple|nsga2|map-elites|cvt|novelty|dns`, `--descriptor gait|body|body-gait`, `--terrain flat|rough|hilly`, `--task forward|target`, `--brain hyperneat|direct`, `--mutation-rate`, `--vary-body-or-brain on|off`, `--protect-new-bodies on|off`, `--leg-damage on|off`, `--import PATH` (a saved robot joins the first generation; repeatable), `--save-best PATH` (the run's best robot as JSON), `--csv PATH`. From code, `symbios_robot_lab::headless::HeadlessLab` runs the same loop.

### Web (WASM)

```bash
cargo install wasm-bindgen-cli --version 0.2.128  # must match the wasm-bindgen pin in Cargo.toml
cargo build --profile wasm-release --target wasm32-unknown-unknown --bin symbios-robot-lab
wasm-bindgen --out-dir ./dist --target web \
  --out-name symbios-robot-lab --no-typescript \
  target/wasm32-unknown-unknown/wasm-release/symbios-robot-lab.wasm
cp index.html ./dist/
```

Serve `./dist/` with any static file server. The deploy ships the module as built: `wasm-opt` shrinks it but makes the compressed download bigger (see `deploy.yml`).

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
4. **Breed**: parents are selected by the active algorithm, and each child varies its body or its brain (see [Breeding](#breeding)); a tenth of each generation is fresh bodies from the morphology presets.

### Brains

Both encodings evolve a NEAT genome and run the same kind of feed-forward network, with one input per sensor value and one `tanh` output per driven joint. An output picks the joint's target position within its range, and the joint's motor servoes there.

- **Direct NEAT**: the genome is the network. Inputs and outputs are numbered, so when crossover or mutation changes the body, the network is resized and the numbers shift meaning.
- **HyperNEAT** (`src/simulation/hyperneat.rs`): the genome is a CPPN, and the network is painted from it for each body, through symbios-neat's `SubstratePainter`. Every input except the clock sits at a point on the body: an IMU's channels around the IMU, a touch or distance sensor at its mount, a hinge's angle at the hinge, a slider's extension at the slider. Each driven joint has an interneuron and a motor neuron where it sits. The CPPN is asked, for each pair of neurons, given both positions and their distance, whether they link and how strongly; and for each joint, its biases and the weights from the clock's sine and cosine (its rhythm's phase and strength). Each kind of link has its own CPPN output, and a fresh CPPN links only neurons near each other, a rule evolution can reshape. Because the brain depends on where things are rather than how they are numbered, an offspring that grows a leg gets a controller for it, and legs placed alike move alike.

Measured with the headless runner (16 to 32 seeds per setting, 30 generations, NSGA-II, 64 robots, 15 s), HyperNEAT reaches a final best fitness of about 1.8 (1.79 to 1.88 with weight limits from 0.25 to 0.6; the lab uses 0.5) against 1.48 for direct NEAT. Two choices made the difference. A fresh CPPN links only neurons near each other: with every link expressed, 84% of fresh motors sat pinned at a limit. And painted weights stay small, so motor neurons follow the clock as smooth waves instead of slamming between their limits: with weights up to 3, HyperNEAT reached only 0.63.

### Breeding

Every genotype carries its body (the L-system, its seed and derivation steps), its brain (the NEAT genome) and its clock's frequency. A child crosses its two parents and mutates the result, but by default only one half of it: half the children cross and mutate the bodies and keep the fitter parent's brain, the other half keep that parent's body and cross and mutate the brains. A new body is first tried with a brain that already walks, and a new brain on a body it was bred for. Measured over 32 seeds with NSGA-II, this beat changing both at once by +0.50 in the pool's best fitness at 30 generations (p = 0.05) and +0.64 at 100, and halved the robots the physics blew apart.

**Protect new bodies** (NSGA-II) adds Cheney et al.'s morphological innovation protection (2018): every body counts the generations it has been in the gene pool, and youth becomes a third objective, so a new body has a few generations to tune its brain before it must outwalk bodies whose brains have been tuned for many. Over 100 generations it kept more and younger bodies (48 distinct in the pool against 36) and closed most of a slower start on varying alone (-0.42 at 30 generations, -0.16 at 100). The paper saw its gains over thousands of generations, so it is an option for long runs and for body diversity.

### Walking to a target

In the target task each brain gets two more inputs, the sine and cosine of the target's bearing from the robot's heading, and the walk scores the ground it gains toward the target instead of forward displacement. Each generation draws one direction, from the run's seed and the generation number, and every robot's target lies 10 m from its spawn point that way: robots of a generation compete on equal footing, and successive generations steer every way. HyperNEAT places the two bearing neurons just ahead of and beside the robot's root part, so a CPPN can paint, say, the left legs' response to a target on the left. In the window, **Click the ground to place the target** sends every robot to one point.

### Leg damage

A leg is the chain of parts from a foot (a part nothing hangs from) up to the first part that branches: the hexapod's six legs of three joints, a millipede's pairs of legs but not its spine, the Scout's neck. Breaking a leg turns off the motors of its driven joints, so it hangs limp within its limits. **Break a leg** in a robot's inspector lames one of its legs at random (any robot, the showcased one included), and **Mend** restores them, in the spirit of Cully et al.'s robots that recover from damage (2015). With **Leg damage** on, every robot walks with one leg broken, a different one per robot and generation (drawn from the run's seed), so evolution favours gaits that survive losing any leg.

### Hall of fame

The best walking robot of every generation is offered to the hall, which keeps the ten best of the session, across restarts. Each can be:

- **watched**: it walks on its own in front of the population, followed by the camera, with the current task's brain inputs. It is never scored and changes nothing in the run (a test checks that a run with a showcased robot repeats one without).
- **copied** or **saved** as JSON: its genotype (body, brain, rhythm) with the fitness and generation it scored.
- **bred**: it joins the next generation (or a restarted run's first) as a body new to that run.

A robot's JSON, or a bare genotype, can be imported by pasting it into the window, by dropping the file onto the window on the desktop, or with `--import` in headless runs. An imported robot must grow a body with driven joints; a direct brain that does not fit its body or the task is resized.

### Quality diversity

MAP-Elites, CVT-MAP-Elites, novelty search and Dominated Novelty Search keep parents that differ, not only the fittest. They measure robots in one of three descriptor spaces, each of two traits scaled to 0..1 (`src/core/descriptor.rs`):

- **Gait**: duty factor (the share of the time the feet are down) and action intensity (how fast the brain moves the joints).
- **Body**: feet (touch sensors, one column of the grid per count up to 12) and length (the longest horizontal extent at rest, up to 6 m). Morphological axes like these keep far more kinds of body alive at equal peak fitness (Nordmoen et al. 2021).
- **Body and gait**: feet and duty factor, how many feet and how much of the time they stand.

Every robot's descriptor holds all four traits, so switching spaces mid-run re-files the archive rather than mixing coordinates. Robots that failed (exploded, too few motors, no progress) never enter an archive.

Dominated Novelty Search (Bahlous-Boldi et al., GECCO 2025) needs no grid or bounds: each robot scores its mean distance to its three nearest fitter robots, and the highest scores survive, so a robot lives by being the best of its region or far from anyone better. The fittest always survives.

Measured over 16 seeds of 30 generations (pool best fitness; distinct bodies kept): MAP-Elites over gait 1.21, 87; Dominated Novelty Search over gait 1.22, 44; MAP-Elites over body 1.66, 45; Dominated Novelty Search over body 1.05, 54. Over body traits many robots share a body exactly (a brain-only child keeps its parent's), and a fitter twin at distance zero leaves Dominated Novelty Search only the best few of each body: more kinds of body, less refinement of each. NSGA-II, which keeps no diversity on purpose, reaches about 2.5 in the same time.

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
| `bevy_symbios_ground` | Terrain height maps, their meshes and heightfield colliders |
| `bevy_symbios` | The `.lsys` asset loader for hot-loading (native only) |

## License

See [LICENSE](LICENSE).

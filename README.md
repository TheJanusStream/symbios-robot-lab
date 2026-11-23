# Symbios Robot Lab

An evolutionary robotics sandbox that co-evolves robot **bodies** (L-System morphologies) and **brains** (NEAT or HyperNEAT neural networks) in a 3D physics simulation.

Built with [Bevy](https://bevyengine.org/) and [Avian3D](https://github.com/Jondolf/avian) physics, on the Symbios crates.

## Features

- **Co-evolution of body and brain**: L-System grammars define the robot's structure (parts, joints, sensors); a neural network drives the joints. Each child varies its body or its brain, and new bodies can be protected while their brains catch up. See [Breeding](#breeding).
- **Three brain encodings**: HyperNEAT (the default) evolves a pattern, a CPPN, that paints the network onto the body, so one genome drives any body; direct NEAT evolves the network itself; pattern generators play each joint a painted rhythm that starts over whenever its foot lands. See [Brains](#brains).
- **Six evolutionary algorithms**: NSGA-II (multi-objective), SimpleGA (elitist tournament), grid MAP-Elites and CVT-MAP-Elites (quality-diversity archives), novelty search, and Dominated Novelty Search (quality diversity without a grid). The quality-diversity algorithms spread robots over their gait, their body, or both. See [Quality diversity](#quality-diversity).
- **Sensors**: IMU (accelerations), touch, ultrasonic range, hinge angles and slider extensions, alongside a clock (its sine and cosine) whose frequency, 0.25 to 4 Hz, each robot evolves, and as an option the body's tilt (see [Sensing tilt](#sensing-tilt)). Each driven joint's output is a target position within its own limits, reached by a motor held to the effort its limit gives (an evolvable constant, capped).
- **Joints**: driven hinges and sliders, passive ball joints with twist and swing limits, and welds.
- **Terrain**: the flat floor, or rough (bumps up to 20 cm) or hilly (hills up to 3 m) ground from [bevy_symbios_ground](https://github.com/TheJanusStream/bevy_symbios_ground), laid afresh each generation from the run's seed and the generation number, so no gait wins by fitting one patch of ground and runs with the same seed walk the same grounds. Heights and bounce are measured above the ground under the body.
- **Two tasks**: walk forward, or walk to targets whose bearing the brain reads, turning both ways between them with an evolved steering reflex (targets drawn anew for every leg of the walk, or one placed by the user in the window). See [Walking to a target](#walking-to-a-target).
- **Anti-"vibrobot" fitness**: forward displacement (or progress toward the target) times six gates (posture, steady commands, body jerk, a minimum speed, foot step cycles, centre-of-mass bounce): a robot that walks keeps nearly nine tenths of its distance, one that slides by vibrating scores nothing. See [Fitness](#fitness).
- **Live UI**: an egui panel with algorithm selection, parameter sliders, fitness graphs, the Pareto front, the quality-diversity archives and the parents' spread. Click a robot to inspect its body, brain and live inputs.
- **Break a leg**: lame any robot from its inspector (its leg's joints go limp) to see how its gait copes, or (experimental) evolve with every robot missing a random leg. See [Leg damage](#leg-damage).
- **Hall of fame**: the best robots, kept between sessions, to watch walking on their own, share as a link, copy or save as JSON, import (pasted, or dropped onto the window on the desktop), or send back into evolution, and a race of the run's generations side by side. See [Hall of fame](#hall-of-fame).
- **Headless runs**: the evolution loop runs without a window, for tests, batch runs and experiments. A seed repeats a whole run exactly.
- **WebAssembly**: deployed to GitHub Pages.

## Getting started

### Native

```bash
cargo run
```

Evolution starts as soon as the window opens; **Restart evolution** starts over. Drop `.lsys` files into `assets/morphologies/` and they join the pool of fresh bodies (native only). Give a share link as an argument (`cargo run -- 'https://...#robot=...'`) and its robot walks as the lab opens.

### Headless

```bash
cargo run --release --example evolve -- --generations 30 --seed 1
```

One row per generation (best and mean fitness, the parent pool's best, the furthest robot, explosions, the pool's distinct bodies and its oldest body's age). Options: `--population`, `--duration`, `--algorithm simple|nsga2|map-elites|cvt|novelty|dns`, `--descriptor gait|body|body-gait`, `--terrain flat|rough|hilly`, `--task forward|target`, `--brain hyperneat|direct|supg`, `--mutation-rate`, `--parent-share F`, `--retests N` (8 by default; see [Breeding](#breeding)), `--vary-body-or-brain on|off`, `--protect-new-bodies on|off`, `--leg-damage on|off`, `--tilt on|off`, `--plans DIR` (breed from the `.lsys` body plans of a folder in place of the lab's own), `--import PATH` (a saved robot joins the first generation; repeatable), `--save-best PATH` (the run's best robot as JSON), `--save-hall PATH` (the hall of fame as JSON), `--fresh N` (at the end, walk the hall's best four again under N fresh conditions, two clones each: their scores may have been lucky; a condition has its own ground, targets and broken legs and sets the clones down somewhere else, up to 30 m either way), `--fresh-spots moved|same` (`same` sets every condition's clones down on the same two spots, as fresh walks did before: on the flat floor with the forward task every condition is then the same two walks, which repeats older figures), `--score PATH` (instead of evolving, score saved robots under fresh conditions; repeatable), `--walks PATH` (with `--fresh` or `--score`: every fresh walk as a CSV row, its score and what the score was made from, with the robot's mass, size, driven joints and feet, to judge robots by other measures than the fitness that bred them), `--csv PATH`. From code, `symbios_robot_lab::headless::HeadlessLab` runs the same loop.

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
  core/                 config, genotypes, population, body plans, share codes,
                        hot-loaded morphologies
  simulation/           SimulationPlugin: bodies, sensors, brains, metrics, fitness,
                        the parent pool and the generational lifecycle, on fixed steps
  main.rs               the windowed app
  ui/                   egui control panel, robot inspector, hall of fame (kept
                        between sessions)
  visuals/              scene, robot meshes and materials, camera follow
examples/evolve.rs      headless runs from the command line
examples/bodyplans.rs   what breeding makes of a body plan, measured
tests/
  headless_evolution.rs whole generations without a window
  evolution_pipeline.rs NEAT mutation, crossover, selection pressure
  morphology_pipeline.rs L-System to blueprint topology, and what the body
                        plans must keep for evolution to build on them
```

### Simulation loop

Every generation lasts a fixed number of physics steps (default 15 simulated seconds at 64 Hz; the window shows them at 5x):

1. **Spawn**: each agent's L-System genotype is derived into a body, and its brain is compiled into a feed-forward network for that body (see [Brains](#brains)).
2. **Evaluate**: every fixed step, sensors fill the brain's inputs, the network drives the joint motors, and metrics accumulate. The first third of the generation is settling and is not measured.
3. **Score**: the walker [fitness](#fitness); NSGA-II adds an economy objective. Robots that explode score below every intact one.
4. **Breed**: parents are selected by the active algorithm, and each child varies its body or its brain (see [Breeding](#breeding)); a tenth of each generation is fresh bodies from the [body plans](#body-plans), and the best parents can walk again in children's places.

### Fitness

A walk's fitness is the distance it covers, forward or toward its target, times six factors between 0 and 1. Each factor is a gate against one way of getting about that is not walking. A robot that walks passes them all near 1 (champions keep nearly nine tenths of their distance), so among walkers the fitness is very nearly the metres walked.

| Gate | Stops |
| --- | --- |
| Posture | tipping and wobbling: the spread of the body's uprightness, 0 at 0.4 |
| Steady commands | vibrating: the joint whose command turns round most often turns more often than a 4 Hz stride; 0 from 8 Hz |
| Body jerk | flailing: the body's own accelerations |
| Minimum speed | creeping at under 0.2 m/s |
| Step cycles | feet that never lift or never land: fewer than three changes per foot |
| Bounce | sliding flat: under 2 cm of rise and fall |

**Steady commands.** A robot can slide across the floor by shaking its joints: with no term against it, the best robot of 3 runs in 16 flipped every command at every evaluation, and one of them slid at 3 m/s on feet that never left the ground. The gate counts how often each joint's command turns round (comes back from the furthest it went by more than a twentieth of the joint's range) and judges the joint that turns most often. A stride is two turns a cycle whatever its size or shape, and a step is one turn at most, so a stride on the fastest clock a brain can have (4 Hz) costs nothing, and from twice that the fitness is 0. Two things the gate does not see: a ripple of a twentieth of a joint's range or less, which the motors (springs of 5 Hz) all but swallow, and when in the walk the turns fall, so a burst of shaking within the walk's budget passes and it is the bounce and step gates that stop a robot sliding on bursts.

**What it replaced.** The fitness used to multiply by two other factors, the mean squared step and the mean squared second difference of the commands, averaged over the joints. Brains run every second step, so the two measured the same thing, and it was the speed of the joints, not vibration: a full-range 1 Hz stride on every joint kept an eighth of its distance, and one full-range step, as a foot sensor asks for, cost a leg nearly all of it. Evolution answered with long levers on few, slow joints. Champions grew from about 250 kg to 649 kg in 100 generations, a leg with a knee or a slider kept 3 to 8 percent of its distance, and a joint that did nothing thinned the average. Judging the worst joint instead of the average made the tax bite harder (champions walked 13.0 m against 19.0 m). Measuring the distance in body sizes, tried as a cure for the growth, changed nothing once the tax was gone (7.3 body sizes walked against 7.5).

**Economy.** NSGA-II's second objective is how little the joints move for the ground covered: the radians every hinge turned through, each counted as one body size of lever, and the metres every slider slid, per metre walked. Neither a body's weight nor its size is a way to score on it. The lab's former estimate of a Cost of Transport (5000 N m for every radian of hinge, over weight times distance) called a heavier body more efficient for the same motion. The Cost of Transport from the work the motors really did (each motor's torque times the angle its hinge turned through, as the physics reports it) is shown in a robot's inspector and is no objective: a joint twitching unloaded costs next to no work, so it does not ask for calm joints.

Measured over 16 seeds of 100 generations with each candidate, each run's best four robots walking again under 8 fresh conditions (the plans start at about 250 kg): with this objective champions walked 23.7 m and weighed 266 kg; with the former estimate 21.9 m and 371 kg; with the motors' real work 16.1 m and 273 kg; and with no second objective at all (SimpleGA) 15.2 m, on five times the joint travel per metre, one fresh walk in eleven ending in a broken robot.

**Measured.** Against the earlier formula with the same body plans, over 16 seeds of 100 generations, each run's best four robots walking again under 8 fresh conditions (10 measured seconds a walk, paired by seed):

| | earlier formula | this one |
| --- | --- | --- |
| Flat floor, distance | 19.0 m | 23.7 m (+4.6, p = 0.07) |
| Flat floor, distance in body sizes | 6.1 | 10.1 (p < 0.01) |
| Rough ground, distance | 14.3 m | 23.7 m (+9.5, p < 0.001) |
| Champion's weight (the plans start at about 250 kg) | 649 kg | 266 kg |
| Furthest robot of generation 30 | 8.6 m | 16.7 m |
| Motor work per walk | 84 kJ | 141 kJ |
| Recorded score kept under fresh conditions | 99% | 89% |
| Robots culled per generation, flat and rough | 2.4 and 1.6 | 3.5 and 1.8 |

The earlier formula's champions stride at their clock's 1 Hz on bodies grown large, and repeat a walk exactly; the present ones are light, stride faster, work harder and vary from walk to walk, which is why the best parents now walk again (see [Breeding](#breeding)). Three of the earlier 16 champions had grown a third pair of legs, none of the present ones on either ground: nothing in the fitness pays for a leg that does not carry the robot further, and the plans' stiff legs do not. What the fitness no longer does is price out legs that could (see [Body plans](#body-plans)).

Scores quoted elsewhere in this README were measured with the earlier formula and compare only with one another.

### Brains

Every encoding evolves a NEAT genome. Two of them run a feed-forward network, with one input per sensor value and one `tanh` output per driven joint; the third plays patterns. An output picks the joint's target position within its range, and the joint's motor servoes there.

- **Direct NEAT**: the genome is the network. Inputs and outputs are numbered, so when crossover or mutation changes the body, the network is resized and the numbers shift meaning.
- **HyperNEAT** (`src/simulation/hyperneat.rs`): the genome is a CPPN, and the network is painted from it for each body, through symbios-neat's `SubstratePainter`. Every input except the clock sits at a point on the body: an IMU's channels around the IMU, a touch or distance sensor at its mount, a hinge's angle at the hinge, a slider's extension at the slider. Each driven joint has an interneuron and a motor neuron where it sits. The CPPN is asked, for each pair of neurons, given both positions and their distance, whether they link and how strongly; and for each joint, its biases and the weights from the clock's sine and cosine (its rhythm's phase and strength). Each kind of link has its own CPPN output, and a fresh CPPN links only neurons near each other, a rule evolution can reshape. Because the brain depends on where things are rather than how they are numbered, an offspring that grows a leg gets a controller for it, and legs placed alike move alike.
- **Pattern generators** (`src/simulation/supg.rs`, after Morse et al. 2013): no network. Each driven joint plays a pattern over time, painted by an evolved CPPN from where the joint sits on the body: given the joint's place and the sine and cosine of its pattern's phase, the CPPN gives the joint's target, and at phase 0 the pattern's offset, which spreads the joints apart (a knee can lag its hip). A pattern lasts one period of the robot's rhythm, and the pattern of a joint on a leg starts over whenever that leg's foot touches down, so legs fall into step with the ground and with each other. Of five encodings Tarapore and Mouret compared (2015), this one evolved best and recovered fastest from damage.

Measured over 32 seeds (30 generations, NSGA-II keeping all 64 parents), the champions of pattern-generator runs walked a little better under fresh conditions than HyperNEAT's (2.13 against 1.86, not significant) and coped best with a broken leg (1.14 against 0.87, p = 0.11); HyperNEAT stays the default. Measured with the headless runner (16 to 32 seeds per setting, 30 generations, NSGA-II, 64 robots, 15 s), HyperNEAT reaches a final best fitness of about 1.8 (1.79 to 1.88 with weight limits from 0.25 to 0.6; the lab uses 0.5) against 1.48 for direct NEAT. Two choices made the difference. A fresh CPPN links only neurons near each other: with every link expressed, 84% of fresh motors sat pinned at a limit. And painted weights stay small, so motor neurons follow the clock as smooth waves instead of slamming between their limits: with weights up to 3, HyperNEAT reached only 0.63.

### Breeding

Every genotype carries its body (the L-system, its seed and derivation steps), its brain (the NEAT genome) and its clock's frequency. A child crosses its two parents and mutates the result, but by default only one half of it: half the children cross and mutate the bodies and keep the fitter parent's brain, the other half keep that parent's body and cross and mutate the brains. A new body is first tried with a brain that already walks, and a new brain on a body it was bred for. Measured over 32 seeds with NSGA-II, this beat changing both at once by +0.50 in the pool's best fitness at 30 generations (p = 0.05) and +0.64 at 100, and halved the robots the physics blew apart.

A child's body takes each symbol's rules from one parent or the other, and each constant whole from one parent or the other. The L-system's own crossover averages the constants, and an average lies between its parents: a robot whose count had just passed a threshold (see [Body plans](#body-plans)), mated with one whose had not, had a child back under it, and every generation drew the population's measures toward their mean. Measured on the same body plans over 16 seeds of 100 generations, champions bred with whole constants walked 24.6 m and with averaged ones 20.0 m (+4.6, p = 0.01). They were also culled more often (6.8 robots a generation against 4.4), which bounding the joints' ranges in the plans brought back to 3.5.

**Parents kept** (NSGA-II, SimpleGA, Dominated Novelty Search) is the share of a generation kept as parents: fewer breed from the best more often, more keep more kinds of body. **Parents walking again** clones the best parents by average score into the next generation, in children's places, and adds each new walk to their average, which the hall of fame shows with the walks behind it. Conditions change between generations (every robot walks from its own spot, and on new ground or toward new targets where asked), so one walk can flatter a robot. Measured under the lab's earlier fitness over 32 seeds with NSGA-II, each run's hall champions walking again as clones under 8 fresh conditions (the share of their recorded score they kept in brackets):

| | flat, 30 generations | flat, 100 | hills, 30 |
| --- | --- | --- | --- |
| all 64 parents kept | 1.86 (82%) | 3.48 (73%) | 1.09 (56%) |
| half kept | 2.29 (81%) | 4.17 (81%) | 1.17 (52%) |
| half kept, 8 walking again | 2.13 (91%) | 3.89 (90%) | 1.41 (76%) |

Keeping half gained +0.43 at 30 generations (p = 0.08) and +0.70 at 100 (p = 0.045), and over 16 seeds it gained with SimpleGA (+0.52) and Dominated Novelty Search (+0.22) too, keeping half as many kinds of body. Half is the default. Walking again kept the hall honest and gained on hills (+0.24 over half alone, p = 0.14), but the children it displaces cost on the flat floor (-0.16 and -0.29, neither significant), and it was off by default.

Under the present [fitness](#fitness) gaits no longer run on the clock alone and a walk's score varies (by 4 to 7 percent for most champions, far more for a few), so 8 parents walk again by default. Over 16 seeds of 100 generations on the flat floor, champions kept 89% of their recorded score under fresh conditions against 70% with none walking again (p = 0.01), broke in 3% of their fresh walks against 9%, and walked as well for it (20.4 against 20.6, failed walks counted as 0).

**Fresh conditions.** A recorded score comes from the walks a robot was selected for, so the figures on this page walk each run's best four robots again under 8 fresh conditions, two clones each. A condition has its own ground, targets and broken legs, and it sets the clones down somewhere else, up to 30 m either way within the square a population starts in. On the flat floor that is all that differs between conditions, and it is enough: a walk depends on the last digit of where it began. Until the conditions moved the spots, every flat-floor condition repeated the same two walks, on two of the very spots evolution sets robots down on, and that flattered. The same 64 champions kept 85.5% of their recorded score on those two spots and 81.5% elsewhere, and broke in 5.5% of their walks there and 9.7% elsewhere (p = 0.01), as often as champions break on rough ground (8.5%, where 82% is kept either way). The flat-floor figures above and under [Fitness](#fitness) are from before, two walks to a robot; arms were compared fairly, on the same two spots, but "score kept" and "walks broken" read better than they were.

**Protect new bodies** (NSGA-II) adds Cheney et al.'s morphological innovation protection (2018): every body counts the generations it has been in the gene pool, and youth becomes a third objective, so a new body has a few generations to tune its brain before it must outwalk bodies whose brains have been tuned for many. Over 100 generations it kept more and younger bodies (48 distinct in the pool against 36) and closed most of a slower start on varying alone (-0.42 at 30 generations, -0.16 at 100). The paper saw its gains over thousands of generations, so it is an option for long runs and for body diversity.

### Walking to a target

In the target task each brain gets two more inputs, the sine and cosine of the target's bearing from the robot's heading, and the walk scores the ground it gains toward its targets instead of forward displacement. While it settles, a robot heads for a target 10 m away in the generation's direction. The measured window is then split into two legs, and at the start of each a new target appears 10 m away at 45 to 135 degrees from the robot's own heading, on alternating sides, so every walk asks for turning both ways. The turns are drawn per generation from the run's seed, so a generation's robots turn alike and compete fairly. HyperNEAT places the two bearing neurons just ahead of and beside the robot's root part, so a CPPN can paint, say, the left legs' response to a target on the left. In the window, **Click the ground to place the target** sends every robot to one point.

Every target-task robot also has a steering reflex (after Lan et al. 2020): each driven joint's output is scaled by `1 + gain x side x turn`, within 0 to 2, where `side` is where the part the joint moves sits across the body (-1 to 1) and `turn` is the target's bearing over pi, so the joints on one side swing shorter and the robot turns. The gain evolves (-2 to 2); a robot bred to walk forward, or imported, gets a random one when the task turns to targets. The brain still reads the bearing and can steer on its own.

The task used to give each generation one target in one direction, and robots scored by walking whichever way they happened to face. Measured over 32 seeds (30 generations), the best robot of each run walking to new targets under 12 fresh conditions: bred on the old task 0.22, bred to walk forward 0.21, bred on this one 0.88 (p < 0.001 against both).

### Sensing tilt

The IMU senses jolts (finite differences of its velocity), not lean. With **Sense tilt** on (`--tilt on`), brains get two more inputs, the vertical components of the body's forward and sideways axes (how far each leans from level), and HyperNEAT places their neurons above the bearing's. Measured over 32 seeds it did not help: -0.07 on hills and -0.18 on the flat floor under fresh conditions (neither significant). It is off by default.

### Leg damage

A leg is the chain of parts from a foot (a part nothing hangs from) up to the first part that branches: the skink's four legs and its tail but not the trunk segment between its leg pairs, the scout's eye stalk. Breaking a leg turns off the motors of its driven joints, so it hangs limp within its limits. **Break a leg** in a robot's inspector lames one of its legs at random (any robot, the showcased one included), and **Mend** restores them, in the spirit of Cully et al.'s robots that recover from damage (2015). With **Leg damage** on, every robot walks with one leg broken, a different one per robot and generation (drawn from the run's seed), in the hope that evolution favours gaits that survive losing any leg. That option is experimental: measured over 32 seeds, the best robots of runs bred with a leg broken walked worse with a broken leg (0.66 under fresh conditions) than robots bred intact (0.87), with pattern generators too (0.78 against 1.14).

### Hall of fame

The best walking robot of every generation is offered to the hall, which keeps the ten best across restarts and, in the window, between sessions: in the browser's local storage, or on the desktop in `symbios-robot-lab/hall_of_fame.json` in the user's data folder (`$XDG_DATA_HOME` or `~/.local/share` on Linux, `~/Library/Application Support` on macOS, `%APPDATA%` on Windows). A kept hall that does not come back whole (a robot the import checks refuse, say) is copied aside first, to `hall_of_fame.unread.json` or a second storage entry, since the next save replaces it. A robot that walks again (see [Breeding](#breeding)) shows its average and the walks behind it. Each can be:

- **watched**: it walks on its own in front of the population, followed by the camera, with the current task's brain inputs. It is never scored and changes nothing in the run (a test checks that a run with showcased robots repeats one without).
- **shared** as a link: the lab's page with the robot in the address, its genotype as compact JSON, deflated and in URL-safe base64 (about 4 KB). Opening the link adds the robot to the hall and sets it walking. It arrives as imported: its score belongs to the run it walked in.
- **copied** or **saved** as JSON: its genotype (body, brain, rhythm) with the fitness and generation it scored.
- **bred**: it joins the next generation (or a restarted run's first) as a body new to that run.

A robot's JSON (or a bare genotype, or a whole saved hall) or a share link can be imported by pasting it into the window, a JSON file by dropping it onto the window on the desktop, or with `--import` in headless runs. An imported robot must grow a body with driven joints and no more than 48 parts, and a direct brain that does not fit its body or the task is resized. It enters the hall once, without the identity number it had where it came from.

A saved champion carries the number of the fitness formula that scored it. One scored by another formula than the lab's present one (a hall kept from an earlier version, say) comes back as an imported robot, without its score: the hall ranks by score, and scores of two formulas do not compare. Its body and brain are untouched; send it into evolution and it earns a score again.

A hall kept from before the [body plans](#body-plans)' rules were last rewritten holds robots written with the former rules. They walk and breed as they did, and a child of one of them and a present plan is a whole body of its first parent's kind. Their line keeps the former rules' ways too: one mutant in seventy lengthens every leg by whole segments.

**Race of generations**: the run keeps its best robot as of generation 0 and every 10th generation since (past 16 kept, the stride doubles and every other one leaves), and **Race** sets up to six of them walking side by side in lanes: generation 0, the latest, and four spread between (all it has while the run is young). The hall shows the standings live, each racer's distance from where it started. Racers, like watched robots, are never scored.

### Quality diversity

MAP-Elites, CVT-MAP-Elites, novelty search and Dominated Novelty Search keep parents that differ, not only the fittest. They measure robots in one of three descriptor spaces, each of two traits scaled to 0..1 (`src/core/descriptor.rs`):

- **Gait**: duty factor (the share of the time the feet are down) and action intensity (how fast the brain moves the joints).
- **Body**: feet (touch sensors, one column of the grid per count up to 12) and length (the longest horizontal extent at rest, up to 6 m). Morphological axes like these keep far more kinds of body alive at equal peak fitness (Nordmoen et al. 2021).
- **Body and gait**: feet and duty factor, how many feet and how much of the time they stand.

Every robot's descriptor holds all four traits, so switching spaces mid-run re-files the archive rather than mixing coordinates. Robots that failed (exploded, too few motors, no progress) never enter an archive.

Dominated Novelty Search (Bahlous-Boldi et al., GECCO 2025) needs no grid or bounds: each robot scores its mean distance to its three nearest fitter robots, and the highest scores survive, so a robot lives by being the best of its region or far from anyone better. The fittest always survives.

Measured over 16 seeds of 30 generations, keeping all 64 parents (pool best fitness; distinct bodies kept): MAP-Elites over gait 1.21, 87; Dominated Novelty Search over gait 1.22, 44; MAP-Elites over body 1.66, 45; Dominated Novelty Search over body 1.05, 54. Over body traits many robots share a body exactly (a brain-only child keeps its parent's), and a fitter twin at distance zero leaves Dominated Novelty Search only the best few of each body: more kinds of body, less refinement of each. NSGA-II, which keeps no diversity on purpose, reaches about 2.5 in the same time. Keeping half the parents, the default since, Dominated Novelty Search over gait keeps half as many kinds of body (22 against 44) and its champions walk a little better under fresh conditions (+0.22, not significant); raise **Parents kept** for diversity.

### Body plans

Generation 0 and each generation's fresh robots draw from three body plans (`src/core/presets.rs`), relatives of one simple body with one motor per leg:

- the **Skink**: a head, two pairs of stiff legs and a tail (a trunk of two segments, six motors);
- the **Scout**: the skink with an ultrasonic eye on a stalk, watching the ground ahead;
- the **Eel**: a trunk of short links on short legs, with a tail tip that slides in and out instead of turning (seven motors, one of them the lab's slider).

They are starting points for evolution, not finished walkers, and they are written for it. One rule grows every segment of every limb: the pair of side limbs the segment bears, the hinge it hangs on, the segment, and the rest of the limb. The trunk is a limb whose segments each bear a pair of legs; a leg is a limb whose own side limbs are told to have no segments. A rule no body uses decays under mutation until it is broken (an earlier draft's rule for a leg with a knee was, in nearly every champion, after 100 generations), and the rule that grows a limb of several segments is the one every trunk uses, so a leg can always become such a limb. A second rule gives the last segment of a leg its foot sensor.

The axiom works out how many segments each limb has, so one mutated constant can change the answer:

- **a knee**: a leg has `round(segs)` segments, one to three, each `leg / that` long. `segs` starts at 1.35; past 1.5 every leg is split into two halves of its old length with a hinge between them, parallel to the hip's, that turns `bend` either way. The leg is as long as it was, hip to foot: a 0.8 m leg becomes two parts of 0.4 m with the knee in the middle. 3.4 percent of the mutations of a body do it;
- **a pair of legs**: the trunk has `round(trunk / link)` segments, one to six, each bearing a pair. 7 percent of the mutations of a body grow a third pair (9 for the eel).

What else the plans keep to:

- every number evolution should tune is a constant the axiom names, the joints' ranges among them (the L-system copies constants into the rules as it reads them, so a constant only the rules use no longer reaches the body when it mutates; a draft whose joint ranges sat in the axiom as plain numbers walked a quarter less far);
- the axiom does the arithmetic and holds the bounds, since mutation never edits it and flips `*` to `/`, `+` to `-` and `>` to `<` in rules but leaves `min`, `max`, `round` and `sqrt` alone: counts are rounded and clamped there, and what the physics cannot bear is bounded there (the trunk no thinner than 0.3 m, a hip turning no further than 0.75 rad either way, which brought the robots culled late in a run back from 11 a generation to 6);
- nothing a body is made of is free: a leg is set out from the trunk by a move that builds no part, so the way from trunk to hip weighs nothing and cannot be seen, and the axiom holds a hip within 0.25 m of where its leg would touch the trunk's flank, where the plans start. Unbounded, that `reach` grew from 0.8 m to 1.5 m and more, and legs hung about a metre of empty air from the body (measured below). A likely reason, which no measurement here isolates: the trunk's joints swing each hip through an arc of that radius;
- the rules compute nothing, not even a limb's count-down: the axiom hands a limb's number of segments down as 10 to the power of 2 to the power of it, and the rule takes a square root for each segment it grows. The count-down used to subtract, and with its sign flipped, which 1.5 percent of the mutations of a body did, every limb grew a whole segment with every step: legs of two, three and four parts, each as long as the whole leg had been;
- a limb ends on a number, not below one: the rule grows while its count is above an end mark the axiom hands down, and a finished limb's count is the mark itself, which is neither above it nor below (the L-system calls numbers equal within a part in a hundred million). Mutation also copies rules, and a copy whose comparison then flipped used to match every finished limb: 41 parts at four steps. Now it matches nothing;
- a condition holds one comparison (the L-system writes an AND back in a form it cannot read again, and the lab stores bodies as written back), and no plan uses a turn together with its opposite, since mutation swaps those only where both occur;
- the numbers that must hold (90 degrees, a 4000 N m motor) are too large for mutation's nudges to move, and the limb rule's own condition compares two parameters and holds no number;
- new rules get new names (`G` and `T` here, `L` and `Y` before): a child takes each rule from one parent or the other by its symbol, so under one name half the children of a robot kept from before and one of these would pair an axiom with a rule that cannot read its counts. As it is, such a child is a whole body of its first parent's kind.

So a leg gains parts only by splitting. Of 60 000 mutants of each plan, 15 to 24 have a chain of more parts than a leg that is longer than one (where mutation set a part of its own into a rule, or swapped two of a rule's modules) against 870 to 900 before; `bodyplans mutate` counts them and says what was rewritten. A robot kept in the hall of fame from before still carries the old rules: sent back into evolution, its line can lengthen legs as it did.

Three things outside the plans let what a mutation grows live: a fresh body is derived with a step more than it needs, so a new pair of legs or a knee is whole at once; a child takes each constant whole from one parent or the other (see [Breeding](#breeding)); and a body may have 48 parts at most, since a mutated rule can grow without end.

**Measured**, over 16 seeds of 100 generations, each run's best four robots walking again under 8 fresh conditions (then still on the same two spots, see [Breeding](#breeding)), the one-rule plans as first written, with a count-down that subtracted, against the plans they replaced (the same three bodies, written with a rule for the trunk and one for the legs, their constants averaged in a child):

| | former plans | one-rule plans |
| --- | --- | --- |
| Flat floor, distance | 23.7 m | 25.8 m (+2.2, p = 0.40) |
| Rough ground, distance | 23.7 m | 22.0 m (-1.7, p = 0.49) |
| Robots culled per generation, flat and rough | 3.5 and 1.8 | 3.5 and 4.8 |
| Champions' fresh walks ending in a broken robot, flat and rough | 3% and 2% | 5% and 8% (rough: p = 0.03) |
| Champions on a third pair of legs | none of 16, on either ground | none of 16, on either ground |
| Champions with a knee | (none possible) | none of 64, on either ground |

So the one-rule catalogue walks as far as the former one, within what 16 runs can tell, and breaks more often on rough ground. It can grow by one mutation what the former needed two for (a pair of legs) or could not grow at all (a knee), and in 100 generations no champion had grown either: one in 16 grew the pair in a draft whose joint ranges were not yet bounded. Most of the distance comes from the breeding, not the grammar: bred with whole constants, the former plans walked 27.5 m on the flat floor (rough ground not run), so under the same breeding the new grammar walks about 2 m less.

The count-down and the way a limb ends were then rewritten as described above, which changes no starting body. Measured the same way, with the fresh conditions on moved spots, against the one-rule plans as first written (their champions walked again on the same moved spots):

| | count-down that could flip | square root and end mark |
| --- | --- | --- |
| Flat floor, distance | 25.7 m | 24.8 m (-0.9, p = 0.75) |
| Rough ground, distance | 22.0 m | 24.5 m (+2.5, p = 0.22) |
| Champions' fresh walks ending in a broken robot, flat and rough | 9.7% and 8.6% | 9.0% and 7.2% |
| Recorded score kept under fresh conditions, flat and rough | 81% and 82% | 84% and 85% |
| Robots culled per generation, flat and rough | 3.5 and 4.8 | 5.0 (p = 0.09) and 4.6 |
| Champions with a knee or a third pair of legs | none of 64, on either ground | none of 64, on either ground |

Closing the ways to longer legs costs no distance that 16 runs can tell. More robots were culled on the flat floor (over the last 30 generations 9.5 a generation against 6.6, p = 0.04, with no difference on rough ground and none in how often champions break); that is unexplained and may be chance.

Last, the hips' reach was bounded. In the champions of the table above it had grown from 0.8 m to a median of 1.45 m on the flat floor and 1.71 m on rough ground (2.3 m at most): between the trunk's surface and a leg's lay 0.95 m and 1.17 m of nothing, where the skink starts with 0.24 m and the eel with 0.14 m. Measured the same way against those plans, with the hip held within 0.25 m of the trunk's flank (these), set on the flank itself, or carried by a welded bar that fills the reach:

| | reach unbounded | these | hip on the flank | welded bar |
| --- | --- | --- | --- | --- |
| Flat floor, distance | 24.8 m | 20.3 m (-4.5, p < 0.01) | 19.0 m | 9.9 m |
| Rough ground, distance | 24.5 m | 14.6 m (-9.9, p < 0.01) | 11.9 m | not run |
| Champions' fresh walks ending in a broken robot, flat and rough | 9.0% and 7.2% | 1.4% and 1.2% (p < 0.01, p = 0.01) | 2.0% and 0% | 0.5% |
| Robots culled per generation, flat and rough | 5.0 and 4.6 | 1.0 and 1.1 | 2.0 and 0.8 | 3.1 |
| Recorded score kept under fresh conditions, flat and rough | 84% and 85% | 91% and 89% | 83% and 80% | 85% |
| Largest gap between trunk and leg | 1.76 m | 0.25 m | none | 0.58 m |
| Best robots with a third pair of legs, flat and rough | none | 1 of 16 and none | 3 of 16 and none | 7 of 16 |

Bounding the reach costs a fifth of the distance on the flat floor and two fifths on rough ground, and the robots then break a sixth as often, one culled a generation where five were. Why they break less is not settled: among the unbounded champions a longer reach did not go with more broken walks (correlations of +0.06 and +0.04), and the bounded ones are also slower and 50 and 25 kg heavier. On rough ground the hip on the flank walks less far than the hip a hand's breadth off it (p < 0.01; on the flat floor 1.2 m less, p = 0.16), and it keeps less of its recorded score. With the bar, the robots of the first generation already reach 1.5 m where they reach 4.4 m without, for a reason not found.

What the bound leaves open: evolution buys a little reach with thickness (on the flat floor the trunk's radius grew by 0.04 m and the legs' by 0.03 m more than without the bound: hips at 0.84 m on average, 0.95 m at most); the bound does not know the head, which the first pair of legs hangs from, so where a head has grown the front legs sit partly inside it; and mutation can still set a move of its own into a rule and lengthen it, which no champion had. A robot kept in the hall of fame keeps the axiom it was bred with, so one kept from before carries its unbounded reach into its line.

What a mutation grows is not kept where it does not pay, and knees do not (measured with the count-down that could flip): bred from a plan that starts with knees, 14 runs in 16 had dropped them after 100 generations, and the two that had not walked 2.3 m and 4.1 m where the others walked 20 m. A leg split by a driven hinge is, with these brains, a worse leg on the flat floor, and on rough ground no champion kept one either.

Two findings from the first catalogue still stand:

- **Plans share a gene pool, and a fast starter takes it over.** A crab on six stiff legs scored twice the skink's fitness after ten generations, won all 16 runs it shared with the skink, and ended them at under half the skink's score. So the plans are one family: any two cross into a body of the same kind.
- **Judge a plan in the mix it ships in, over 100 generations.** `cargo run --release --example evolve -- --plans DIR` breeds from the `.lsys` files of a folder in place of the lab's plans.

`cargo run --release --example bodyplans -- mutate` measures what breeding makes of each plan (or of any `.lsys` file given with `--file`): how many mutants keep their build, change it or are lost, what the changed ones become, and how many have a leg of more parts, as long as before (a split) or longer. Its other modes list a plan's parts and joints, cross plans, walk mutants to see which the physics culls, and print what a saved robot's fitness is made of. On the desktop, `assets/morphologies/example_walker.lsys` joins the three plans: a copy of the skink to edit.

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

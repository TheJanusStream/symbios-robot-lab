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

One row per generation (best and mean fitness, the parent pool's best, the furthest robot, explosions, the pool's distinct bodies and its oldest body's age). Options: `--population`, `--duration`, `--algorithm simple|nsga2|map-elites|cvt|novelty|dns`, `--descriptor gait|body|body-gait`, `--terrain flat|rough|hilly`, `--task forward|target`, `--brain hyperneat|direct|supg`, `--mutation-rate`, `--parent-share F`, `--retests N` (8 by default; see [Breeding](#breeding)), `--vary-body-or-brain on|off`, `--protect-new-bodies on|off`, `--leg-damage on|off`, `--tilt on|off`, `--import PATH` (a saved robot joins the first generation; repeatable), `--save-best PATH` (the run's best robot as JSON), `--save-hall PATH` (the hall of fame as JSON), `--fresh N` (at the end, walk the hall's best four again under N fresh conditions: their scores may have been lucky), `--score PATH` (instead of evolving, score saved robots under fresh conditions; repeatable), `--walks PATH` (with `--fresh` or `--score`: every fresh walk as a CSV row, its score and what the score was made from, with the robot's mass, size, driven joints and feet, to judge robots by other measures than the fitness that bred them), `--csv PATH`. From code, `symbios_robot_lab::headless::HeadlessLab` runs the same loop.

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

The earlier formula's champions stride at their clock's 1 Hz on bodies grown large, and repeat a walk exactly; the present ones are light, stride faster, work harder and vary from walk to walk, which is why the best parents now walk again (see [Breeding](#breeding)). Three of the earlier 16 champions had grown a third pair of legs, none of the present ones on the flat floor and one on rough ground: nothing in the fitness pays for a leg that does not carry the robot further, and the plans' stiff legs do not. What the fitness no longer does is price out legs that could (see [Body plans](#body-plans)).

Scores quoted elsewhere in this README were measured with the earlier formula and compare only with one another.

### Brains

Every encoding evolves a NEAT genome. Two of them run a feed-forward network, with one input per sensor value and one `tanh` output per driven joint; the third plays patterns. An output picks the joint's target position within its range, and the joint's motor servoes there.

- **Direct NEAT**: the genome is the network. Inputs and outputs are numbered, so when crossover or mutation changes the body, the network is resized and the numbers shift meaning.
- **HyperNEAT** (`src/simulation/hyperneat.rs`): the genome is a CPPN, and the network is painted from it for each body, through symbios-neat's `SubstratePainter`. Every input except the clock sits at a point on the body: an IMU's channels around the IMU, a touch or distance sensor at its mount, a hinge's angle at the hinge, a slider's extension at the slider. Each driven joint has an interneuron and a motor neuron where it sits. The CPPN is asked, for each pair of neurons, given both positions and their distance, whether they link and how strongly; and for each joint, its biases and the weights from the clock's sine and cosine (its rhythm's phase and strength). Each kind of link has its own CPPN output, and a fresh CPPN links only neurons near each other, a rule evolution can reshape. Because the brain depends on where things are rather than how they are numbered, an offspring that grows a leg gets a controller for it, and legs placed alike move alike.
- **Pattern generators** (`src/simulation/supg.rs`, after Morse et al. 2013): no network. Each driven joint plays a pattern over time, painted by an evolved CPPN from where the joint sits on the body: given the joint's place and the sine and cosine of its pattern's phase, the CPPN gives the joint's target, and at phase 0 the pattern's offset, which spreads the joints apart (a knee can lag its hip). A pattern lasts one period of the robot's rhythm, and the pattern of a joint on a leg starts over whenever that leg's foot touches down, so legs fall into step with the ground and with each other. Of five encodings Tarapore and Mouret compared (2015), this one evolved best and recovered fastest from damage.

Measured over 32 seeds (30 generations, NSGA-II keeping all 64 parents), the champions of pattern-generator runs walked a little better under fresh conditions than HyperNEAT's (2.13 against 1.86, not significant) and coped best with a broken leg (1.14 against 0.87, p = 0.11); HyperNEAT stays the default. Measured with the headless runner (16 to 32 seeds per setting, 30 generations, NSGA-II, 64 robots, 15 s), HyperNEAT reaches a final best fitness of about 1.8 (1.79 to 1.88 with weight limits from 0.25 to 0.6; the lab uses 0.5) against 1.48 for direct NEAT. Two choices made the difference. A fresh CPPN links only neurons near each other: with every link expressed, 84% of fresh motors sat pinned at a limit. And painted weights stay small, so motor neurons follow the clock as smooth waves instead of slamming between their limits: with weights up to 3, HyperNEAT reached only 0.63.

### Breeding

Every genotype carries its body (the L-system, its seed and derivation steps), its brain (the NEAT genome) and its clock's frequency. A child crosses its two parents and mutates the result, but by default only one half of it: half the children cross and mutate the bodies and keep the fitter parent's brain, the other half keep that parent's body and cross and mutate the brains. A new body is first tried with a brain that already walks, and a new brain on a body it was bred for. Measured over 32 seeds with NSGA-II, this beat changing both at once by +0.50 in the pool's best fitness at 30 generations (p = 0.05) and +0.64 at 100, and halved the robots the physics blew apart.

**Parents kept** (NSGA-II, SimpleGA, Dominated Novelty Search) is the share of a generation kept as parents: fewer breed from the best more often, more keep more kinds of body. **Parents walking again** clones the best parents by average score into the next generation, in children's places, and adds each new walk to their average, which the hall of fame shows with the walks behind it. Conditions change between generations (every robot walks from its own spot, and on new ground or toward new targets where asked), so one walk can flatter a robot. Measured under the lab's earlier fitness over 32 seeds with NSGA-II, each run's hall champions walking again as clones under 8 fresh conditions (the share of their recorded score they kept in brackets):

| | flat, 30 generations | flat, 100 | hills, 30 |
| --- | --- | --- | --- |
| all 64 parents kept | 1.86 (82%) | 3.48 (73%) | 1.09 (56%) |
| half kept | 2.29 (81%) | 4.17 (81%) | 1.17 (52%) |
| half kept, 8 walking again | 2.13 (91%) | 3.89 (90%) | 1.41 (76%) |

Keeping half gained +0.43 at 30 generations (p = 0.08) and +0.70 at 100 (p = 0.045), and over 16 seeds it gained with SimpleGA (+0.52) and Dominated Novelty Search (+0.22) too, keeping half as many kinds of body. Half is the default. Walking again kept the hall honest and gained on hills (+0.24 over half alone, p = 0.14), but the children it displaces cost on the flat floor (-0.16 and -0.29, neither significant), and it was off by default.

Under the present [fitness](#fitness) gaits no longer run on the clock alone and a walk's score varies (by 4 to 7 percent for most champions, far more for a few), so 8 parents walk again by default. Over 16 seeds of 100 generations on the flat floor, champions kept 89% of their recorded score under fresh conditions against 70% with none walking again (p = 0.01), broke in 3% of their fresh walks against 9%, and walked as well for it (20.4 against 20.6, failed walks counted as 0).

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

A robot's JSON (or a bare genotype, or a whole saved hall) or a share link can be imported by pasting it into the window, a JSON file by dropping it onto the window on the desktop, or with `--import` in headless runs. An imported robot must grow a body with driven joints, and a direct brain that does not fit its body or the task is resized. It enters the hall once, without the identity number it had where it came from.

A saved champion carries the number of the fitness formula that scored it. One scored by another formula than the lab's present one (a hall kept from an earlier version, say) comes back as an imported robot, without its score: the hall ranks by score, and scores of two formulas do not compare. Its body and brain are untouched; send it into evolution and it earns a score again.

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

- the **Skink**: a head, two pairs of stiff legs and a tail of two hinged segments (six motors);
- the **Scout**: the skink with an ultrasonic eye on a stalk, watching the ground ahead;
- the **Eel**: a long trunk of short links on short legs, whose tail slides in and out instead of turning (seven motors, one of them the lab's slider).

They are starting points for evolution, not finished walkers, and they are written for it:

- every number evolution should tune is a constant the axiom names (the L-system copies constants into the rules as it reads them, so a constant only the rules use no longer reaches the body when it mutates);
- legs and trunk segments come from rules, never from the axiom, which mutation cannot edit;
- one rule grows both legs of a pair, so what mutation does to a leg it does to both;
- the trunk gains a segment each time `trunk / link` passes 2.5, 3.5 and so on, each plan starts a tenth below such a threshold, and the new segment's legs grow with one more derivation step;
- rules compute nothing but the trunk's count-down, since mutation flips `*` to `/` and `+` to `-` wherever it finds them, and no plan uses a turn together with its opposite, since mutation swaps those only where both occur;
- the numbers that must hold (90 degrees, a threshold of 150) are too large for mutation's nudges to move.

Measured under the lab's earlier fitness over 16 seeds of 100 generations, by each run's best four robots walking again under 8 fresh conditions: 10.84 against 3.89 for the former catalogue of eight hand-built walkers (+6.94 paired by seed), and on rough ground 7.69 against 2.92. In 3 of the 16 runs the best robot had grown a third or a fourth pair of legs on a longer trunk; nine of the former catalogue's 16 runs ended on a body cut down to five or six motors. As many robots are culled as before (2.4 per generation), most of them late in a run.

Two findings shaped the catalogue more than any grammar, both from trials with earlier drafts of these plans:

- **The fitness of the time paid for few motors.** A leg with a knee or a slider near its foot sensor is steered by that sensor in steps, and the tax the fitness then put on joint speed (see [Fitness](#fitness)) left such legs 3 to 8 percent of their distance. Bodies with one motor per leg learned to walk in a few generations, so the plans have stiff legs, and what evolution adds is legs and trunk. The present fitness no longer prices such legs out: bred alone for twelve generations, the drafts with kneed and with sliding legs score 5.9 and 6.1 where the plans score 8.2 (0.26 and 0.24 against 1.6 before), so they are worth another look.
- **Plans share a gene pool, and a fast starter takes it over.** A crab on six stiff legs scored twice the skink's fitness after ten generations, won all 16 runs it shared with the skink, and ended them at 4.67. A seal on two legs with knees won 15 of 16 and ended at 3.48. So the plans are one family: any two cross into a body on the same stiff legs.

`cargo run --release --example bodyplans -- mutate` measures what breeding makes of each plan (or of any `.lsys` file given with `--file`): how many mutants keep their build, change it or are lost. Its other modes list a plan's parts and joints, cross plans, walk mutants to see which the physics culls, and print what a saved robot's fitness is made of. On the desktop, `assets/morphologies/example_walker.lsys` joins the three plans: a copy of the skink to edit.

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

# Direction permutation diagnostic

The registered `shuffled` control is a one-bin(10degree) cyclic rotation. Adjacent structural rays are correlated, so this is a weak angular misalignment test, not a random destruction of directional structure. Preserve it and label it accurately.

Add a fixed, GT-independent permutation of all36directions(default_rng20261109), identical across scenes/models, on fixed pure visual Top20. Evaluate raw/chirp/embedding/attention36ray outputs, bothseeds/all12heldscenes, uniform/uncertainty/pair-selection and geometry/visual fusion. No fitting or coefficient selection. This supplements the original fullbank10degree rotation; no random-permutation fullbank claim is made. Its role is falsification of direction-specific evidence, not model selection or independent validation.

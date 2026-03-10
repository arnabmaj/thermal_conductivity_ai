# Feature Design for Thermal Conductivity Prediction

## 1. Physical Factors Controlling Thermal Conductivity

Thermal conductivity in crystals is governed by several key physical factors.

### Phonon group velocity
### Phonon scattering rates
### Anharmonicity
### Atomic mass
### Bond strength
### Crystal structure

---

## 2. Simple Composition-Based Features

These are features derived directly from composition.

- average atomic mass
- atomic mass variance
- average electronegativity
- atomic radius statistics
- number of atoms in unit cell

---

## 3. Structure-Based Features

These depend on crystal structure.

- density
- lattice parameters
- coordination numbers
- packing fraction

---

## 4. Pressure-Dependent Features

High pressure affects:

- density
- bonding strength
- phonon frequencies
- anharmonicity

Features could include:

- pressure
- pressure-induced density change
- bulk modulus

---

## 5. Advanced Phonon Features (Future)

These require phonon calculations.

- phonon group velocities
- phonon lifetimes
- Grüneisen parameters
- phonon density of states

---

## 6. Minimal Feature Set (Initial Model)

The first ML model will use a minimal feature set:

- composition descriptors
- atomic mass statistics
- electronegativity statistics
- pressure
- density

More advanced features may be added later.
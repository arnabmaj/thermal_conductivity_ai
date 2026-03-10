# Problem Statement

## 1. Project Title
Explainable AI for Predicting Thermal Conductivity of Crystals at High Pressure

## 2. Core Problem
This project aims to predict the lattice thermal conductivity of crystalline materials under high pressure using machine learning. In addition to prediction, the project will build an AI system that explains why the thermal conductivity is high or low using both model-derived features and scientific literature. The long-term goal is to create a scientifically grounded and explainable research assistant for thermal transport under pressure.

## 3. Why This Matters
Thermal conductivity is a fundamental property that governs heat transport in solids. Under high pressure, phonon spectra, anharmonicity, bonding, and structural stability can change significantly, which may strongly affect thermal transport. Understanding thermal conductivity at pressure is important for condensed matter physics, geophysics, planetary materials, and materials design. A predictive and explainable AI framework could accelerate both scientific understanding and materials screening.

## 4. Main Research Question
Can machine learning predict the thermal conductivity of crystals at high pressure with useful accuracy, while also providing physically meaningful explanations grounded in scientific literature?

## 5. Secondary Research Questions
- Which material descriptors are most important for predicting thermal conductivity under pressure?
- How does pressure influence the physical factors controlling thermal transport?
- Can a retrieval-augmented AI system explain thermal conductivity trends using published scientific knowledge?

## 6. Project Objective
The objective is to build an AI system that takes a crystal and pressure condition as input, predicts thermal conductivity, retrieves relevant scientific literature, and generates a physically grounded explanation of the prediction.

## 7. Inputs and Outputs
**Inputs**
- Crystal/composition/structure information
- Pressure
- Optional derived physical descriptors

**Outputs**
- Predicted thermal conductivity
- Explanation of why thermal conductivity is high or low
- Supporting scientific references from literature

## 8. High-Level System Flow
Material + pressure information will be converted into descriptors and passed to a machine learning model to predict thermal conductivity. The system will then use feature importance and scientific literature retrieval to identify the likely physical reasons behind the prediction. A language model will combine these signals into a structured scientific explanation or report.

## 9. Academic Goal
The academic goal is to develop a publishable framework for explainable prediction of thermal conductivity under pressure. A paper may emerge if the project demonstrates useful predictive performance, physically meaningful interpretation, and possibly screening or hypothesis generation for promising materials.

## 10. Industry/Engineering Goal
This project is also designed to demonstrate modern AI engineering skills, including machine learning, retrieval-augmented generation, explainable AI, API design, Docker, CI/CD, AWS deployment, and reproducible project structure using GitHub.
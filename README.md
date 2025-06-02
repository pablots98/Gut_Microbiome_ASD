<h1 align="center">Gut Microbiome in Autism Spectrum Disorder (ASD)</h1>

<p align="center">
  <img src="https://img.shields.io/badge/status-active-brightgreen" alt="project-status">
  <img src="https://img.shields.io/github/license/pablots98/Gut_Microbiome_ASD" alt="license">
  <img src="https://img.shields.io/badge/analysis-QIIME2%20%2B%20R-blueviolet" alt="stack">
</p>

> **Reproducible pipeline** for importing 16S rRNA reads, processing them with QIIME 2, running statistical analyses in R (Phyloseq/DESeq2) and visualising results that explore the link between the gut microbiome and ASD.

---

## Table of Contents
1. [Scientific background](#scientific-background)  
2. [Repository structure](#repository-structure)  
3. [Requirements](#requirements)  
4. [Installation](#installation)  
5. [Quick start](#quick-start)  
6. [Detailed workflow](#detailed-workflow)  
7. [Example results](#example-results)  
8. [How to cite](#how-to-cite)  
9. [Contributing](#contributing)  
10. [License](#license)  
11. [References](#references)  

---

## Scientific background
Multiple studies report consistent differences in the composition and function of the gut microbiome of individuals with ASD compared with neurotypical controls, suggesting a role in pathophysiology and as a potential therapeutic target.  
This project implements a fully reproducible workflow to investigate such patterns from public or in-house 16S rRNA data.


## Requirements
* **Conda ≥ 4.14** (or Mamba)  
* **QIIME 2 2024.10**  
* **R ≥ 4.3** with `phyloseq`, `microbiome`, `DESeq2`, `ggplot2`, `tidyverse`  
* Or **Docker ≥ 24** (image provided)

## Installation
```bash
# Clone the repository
git clone https://github.com/pablots98/Gut_Microbiome_ASD.git
cd Gut_Microbiome_ASD

# Option A – Conda
conda env create -f envs/environment.yml
conda activate gut_asd

# Option B – Docker
docker build -t gut_micro_asd .
```

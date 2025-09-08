# Introduction to GNU Radio using GRC and Python

A comprehensive workshop for learning GNU Radio through hands-on examples using GNU Radio Companion (GRC) and Python programming.

## 🎯 Workshop Objectives

By the end of this workshop, participants will be able to:
- Understand Software Defined Radio (SDR) concepts and applications
- Build signal processing flowgraphs using GNU Radio Companion
- Create custom Python blocks for GNU Radio
- Implement basic analog and digital communication systems
- Interface GNU Radio with hardware SDR devices

## 📚 Prerequisites

- Basic understanding of signal processing concepts
- Familiarity with Python programming
- GNU Radio 3.10+ installed (see installation guide in `resources/installation/`)

## 🗂️ Repository Structure

```
├── slides/          # LaTeX presentation materials
├── examples/        # Working GNU Radio examples
│   ├── grc/         # GNU Radio Companion flowgraphs
│   └── python/      # Python-based implementations
```

## 🚀 Getting Started

1. **Build Slides:**
   ```bash
   cd slides
   make
   ```

2. **Run First Example:**
   ```bash
   gnuradio-companion examples/grc/01_basics/signal_source.grc
   ```

## 📋 Workshop Modules

### Module 1: Introduction to Software Defined Radio
- SDR fundamentals and architecture
- IQ data and complex signals
- Sampling theory and bandwidth

### Module 2: GNU Radio Companion Basics
- Interface overview and workflow
- Signal sources and sinks
- Data types and sample rates
- Streams vs PDUs

### Module 3: Python Integration
- Converting GRC to Python
- Creating custom blocks
- Embedding GNU Radio in applications

### Module 4: Signal Processing & Communications
- Filters and decimation
- Analog modulation (AM/FM)
- Digital modulation (OOK/FSK/QAM)
- Packet radio basics

### Module 5: Hardware Integration
- RTL-SDR configuration
- USRP setup
- Network interfaces (TCP/UDP/ZMQ)

### Module 6: Out-of-Tree Modules
- Creating reusable modules
- Using gr_modtool
- Packaging and distribution

## 🛠️ Required Software

- GNU Radio 3.10 or later
- Python 3.8+
- NumPy, SciPy, Matplotlib
- LaTeX distribution (for slides)
- Optional: gr-osmosdr (for RTL-SDR support)

## 📖 Additional Resources

- [GNU Radio Documentation](https://www.gnuradio.org/doc/)
- [GNU Radio Tutorials](https://wiki.gnuradio.org/index.php/Tutorials)


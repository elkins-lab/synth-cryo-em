# Scientific Foundations

`synth-cryo-em` is built on standard physical models used in structural biology and electron microscopy.

## Density Generation

The mapping from atomic coordinates to 3D density is performed using [Gemmi](https://gemmi.readthedocs.io/).

### Resolution and Blur
In structural biology, the relationship between resolution ($d$) and the isotropic B-factor ($B$) is often approximated as:
$$B \approx 8 d^2$$
This ensures that the information content is significantly attenuated beyond the target resolution. When `--no-bfactors` (default) is used, a global blur is applied to all atoms using this relationship.

### Local Resolution
When `--bfactors` is enabled, the atomic B-factors from the input PDB/CIF file are used directly. This allows for simulating "local resolution" effects where flexible regions of a protein appear more blurred than the rigid core.

## Contrast Transfer Function (CTF)

The simulation of the microscope's optics follows the standard phase-contrast model. The phase shift $\chi(k)$ at spatial frequency $k$ is given by:
$$\chi(k) = \pi \lambda k^2 (\Delta f - 0.5 \lambda^2 k^2 C_s)$$
where:
- $\lambda$ is the relativistic electron wavelength.
- $\Delta f$ is the defocus.
- $C_s$ is the spherical aberration.

The resulting CTF is:
$$CTF(k) = -(\sqrt{1-w^2} \sin(\chi(k)) + w \cos(\chi(k)))$$
where $w$ is the amplitude contrast.

### Envelope Function
An optional envelope function can be applied to simulate spatial and temporal incoherence:
$$E(k) = e^{-B_{env} k^2 / 4}$$

## Validation Metrics

### Fourier Shell Correlation (FSC)
FSC measures the normalized cross-correlation between two 3D maps in shells of constant spatial frequency. It is the standard tool for assessing resolution in Cryo-EM.
$$FSC(k) = \frac{\sum_{|k'|=k} F_1(k') \cdot F_2(k')^*}{\sqrt{\sum_{|k'|=k} |F_1(k')|^2 \cdot \sum_{|k'|=k} |F_2(k')|^2}}$$

### Cross-Correlation Coefficient (CCC)
CCC provides a global measure of similarity between two maps in real space:
$$CCC = \frac{\sum (M_1 - \bar{M_1})(M_2 - \bar{M_2})}{\sqrt{\sum (M_1 - \bar{M_1})^2 \sum (M_2 - \bar{M_2})^2}}$$

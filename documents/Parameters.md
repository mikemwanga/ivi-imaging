## Parameters/adjustments for workflow

- Pre-processing (maybe for cell masks instead, excluding background)
  
  - Denoising --> no
  - Other filters... --> no
  - Deconvolution ? --> Ben: don't do it...

- Normalization across samples -> need controls! (otherwise arbitrary...)
  
  - Possible to acquire **background signal** within sample (e.g. remove cells from one half) ???

- Segmentation --> only if not working properly

- **Dilation/erosion** for mean --> try...

- Alternatives to mean signal (median, functions, filters...) --> mean = usual --> try **percentiles!**

- Threshold for binning --> use mocks (avg_ctrl + 3 * sd_ctrl) or OTSU

- (>2 bins; evtl. with +/- around threshold)



Alternative approaches:

- **Convpaint** ? Then classify cells given the ratios of semantic classes? Or just use to remove debris ...
- Or **train classifier** on cells directly? --> Ana checks her old script...



Mit Ben:
für signal: normal = mean; aber evtl. auch mal upper_quartile o.ä. probieren (allg. Percentiles...)
mid_slice statt projection ? --> aber vermutlich auch nicht repräsentativer

--> for cell signals (esp. goblets): optimize "mode" (percentile/median and maybe even overlap?) and maybe also use dilation/erosion...

--> decide for concept of thresholding, especially infection signal
Project Title- Online Image Authentication using Transform based Watermarking Techniques

Objective-
    -To create an advanced algorithm in order to withstand the tamperng of the digtal images using transform based techniques le DCT and DWT.
    -To enhance the security of digital images by maintaining the robustness and imperceptibility of the digital images.

DWT-Discrete Wavelet Transform

     The Discrete Wavelet Transform (DWT) is extensively used in signal and image processing for multi-resolution analysis of signals For a general multi-level DWT, sub-bands can be labeled as follows:
    • LL (Approximation): Low-frequency components.
    • LH (Horizontal Detail): High-frequency details in the horizontal direction.
    • HL (Vertical Detail): High-frequency details in the vertical direction.
    • HH (Diagonal Detail): High-frequency details in the diagonal direction.
    In a two-level DWT, these sub-bands further decompose, resulting in sub-sub-bands
    (e.g., LL2, LH2, HL2, HH2) representing components after the second-level DWT. This
    hierarchical structure continues for additional levels, providing a multi-resolution representation
    of the original signal, with each sub-band containing information at various
    scales and orientations.

DCT-Discrete Cosine Transform

     The Discrete Cosine Transform (DCT) is a mathematical transformation widely used in signal processing and image compression. The DCT transforms a sequence of values into a set of cosine functions with different frequencies. 
     It is closely related to the Discrete Fourier Transform (DFT) but uses only real numbers and is more computationally efficient for symmetric signals. The DCT is commonly used in applications like image and signal compression, where it
     transforms a sequence or image into a set of coefficients, allowing many of these coefficients to be discarded to achieve compression while retaining essential information.
   

Methodology-

   EMBEDDING METHODOLOGY


    This algorithm securely embeds data into the host medium, ensuring minimal distortion and high integrity.
    The following steps are involved in embedding verifiable URL to the images:
    1. Obtain the url that verifies the authenticity of the certificate.
    2. Convert the url to image
    3. Apply DCT to the image to obtain a 2D matrix
    4. The coefficients below the secondary diagonal of the DCT matrix are made zero since these elements have less significance than the elements above the secondary diagonal.
    5. Obtain the 2-level DWT of the cover image-certificate
    6. Embed the DCT of the url-image to the DWT.
    7. Obtain 2 Level IDWT of the resultant image

   EXTRACTION METHODOLOGY


    This algorithm accurately retrieves embedded data from the host, maintaining its original integrity and ensuring efficient extraction.
    The online image authentication involves the following steps:
    1. Obtain the watermarked Image-certificate
    2. From the 2-level DWT of the watermarked image, extract the HH2 component.
    3. Reduce the element values below the secondary diagonal to zero.
    4. Using inverse DCT, obtain the image containing the verification link (url)
    5. Extract the verification link and verify the document authenticity.



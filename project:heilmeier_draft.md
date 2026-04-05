Heilmeier Questions

Question#1 What are you trying to do with the new design?

Answer: Improve and simplify the implementation of the Radial Basis Kernel for use in SVM and other machine learning applications. We suggest Horner's approximation of the Hadamard Expansion of the Gaussian Kernel Matrix with K=12 to approximate the RBF kernel.

Question#2 How is the RBF kernel used today and what are the limits of it current use?

Answer: In many cases such as in Embedded Systems, like wearable health monitors for ECG/EEG which operate on a limited power budget, KSVM inference is not practical. They could become practical with a Horner chip. Also, the algorithm inside IoT sensors and Smart home devices that incorporate on-device pattern recognition could be greatly improved.

Question#3 What is the new approach and what difference will it make?

Answer: It will allow machine learning algorithms to incorporate RBF kernel inference which is a standard for many machine learning applications into systems that have little or no CPU support and/or power consumption restrictions that make this type of inference to costly with standard methods. Also having a fixed latency will provide more reliable reaction times for systems such as autononous vehicles that use classification algorithms in real-time. The computational latency is currently 60-120 cycles for current FPGA programs that use CORDIC lookup tables whereas it would be 12 cycles with a K=12 Horner chip.
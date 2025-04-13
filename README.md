## **Handwriting Analysis Project**
Python pipeline created to extract useful metrics from handwritten text. Developed as an experiment to measure how the time spent writing affects various handwriting characteristics. The code is general-purpose, and can be adapted to analyze other independent variables as well.

### **Dependent Variables**
**Area** - The number of pixels within a continous shape (Letters or clumps of letters)

**Size** - The size of a minimum fitting rectangle around each continous shape (in # of pixels)

**Spacing** - The distance between each continous shape

**Curvature** - Using an algorithm the approximates the contour of a shape, the curvature is the ratio of how compressed the new approximation is to the original. In a sense, it measures the complexity or curvature of the shape

**Readability** - Using pytessearct to interpret the written text, it is able to act as a consistent readibility grader based on how many sequences of 2 or more characters it gets correct


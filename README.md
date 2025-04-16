# **Handwriting Analysis Project**
Python pipeline created to extract useful metrics from handwritten text. Developed as an experiment to measure how the time spent writing affects various handwriting characteristics. The code is general-purpose, and can be adapted to analyze other independent variables as well.

## **Dependent Variables**
**Continous Lines** - How many continous lines/shapes there are in the written text. A lower value would indicate letters are more connected

**Area** - The number of pixels within a continous shape (Letters or clumps of letters)

**Size** - The size of a minimum fitting rectangle around each continous shape (in # of pixels). A higher score indiactes the letter or groups of letters take up more space.

**Area Size Ratio** - A ratio of how much of the minimum fitting rectangle is filled. May be an indicator of line thickness.

**Spacing** - The distance between each continous shape. A higher score indicates the distance between letters, or groups of letters are farther apart.

**Curvature** - Using an algorithm the approximates the contour of a shape, the curvature is the ratio of how compressed the new approximation is to the original. In a sense, it measures the complexity or curvature of the shape. A higher score indicates higher complexity or curvature.

**Readability** - A measure of what the written text was interpreted as compared to what it should be. It uses PyTessearct as a consistent "reader", and compares the string that it extracts to the correct string using the levenshtein distance algorithm. This algorithm functions based on how many "edits" you need to make to align two strings, and calculates readibility by dividing the number of edits made by the number of characters in the original sentence. It is capped between 0 and 1, and a higher score indicates better readability.

## **Experiment Setup**
The goal of our experiment was to investigate how the time we spend writing a sentence influences certain handwriting characteristics. Our indepdent variables involved writing fast, normal, and slow (or carefully), as well with a stylus, pencil, and pen. We collected 10 samples per indepedent variable combination, for a total of 90 samples each. We were only allowed to collect data from the members of our group, so we collected 270 samples total.

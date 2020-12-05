# covid_ldpcp
LDPCP (Local Directional Pattern CoroNet Descriptor) Method for identifying COVID cases from Chest X-ray images

Involves 3 folders that form a cohesive feature extraction and classification system: Dataset, Feature_Extract and SVM

### Dataset Folder

Contains functions for creating, analyzing, processing and augmenting the dataset being utilized.
More details of the functions given in /Dataset/info.txt (To be added)
/Dataset/config.py needs to be updated with the relevant details for use

The dataset created for the classification purpose is given in the following github repository:
<to_be_added>

### Feature_Extract Folder

Contains functions for extracting the dataset for neural network use, and the neural networks that were used for feature extraction.
The features are saved in a MYSQL database
Also contains some utility functions for getting training statistics of the neural network performance.
More details of the functions given in /Feature_Extract/info.txt (To be added)
/Feature_Extract/config.py needs to be updated with the relevant details for use

### SVM Folder

Contains functions for extracting the saved features from MySQL, and also the SVM for final classification.
More details of the functions given in /SVM/info.txt (To be added)
/SVM/config.py needs to be updated with the relevant details for use
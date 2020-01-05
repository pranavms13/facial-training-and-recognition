# facial-training-and-recognition

This is a simple python code for facial training and recognition using OpenCV.


### Software Requirements

- Numpy
- cv2
- Image from PIL
- argparse

### Hardware Requirements
- Computer with good Camera 😂.

## Usage

- Clone the Repository or Download the zip and extract.
- Install all the Requirements by running,
```pip install -r requirements.txt```
- First run the capturer.py program to capture the samples of face with parameters ``-n <person name> and -c <no of samples>``. 
- **For example**, ```python capturer.py -n pranav -c 50```
 This captures 50 Samples under the name "pranav".
 - Next we need to run the trainer. To do so, we run ```python faces_train.py```
 Doing so produces a trained model "trainer.yml" and a pickle file containing the id of all the captured people.
 - After training, we can finally Run the Recognition Program. This makes use of the trained model and predicts the person. To start prediction we run, ```python face.py```

#### Donations to

-   UPI :  `pranavms13@apl`
-   XMR :  `84B2PEVJjenN31h8HcnY4uCFPVcicxvVCAJAYVeYnPLoHkaVGHHWpVxCm6Gn9beEir2CjffgEoXtiDjpAZCmy4ap6uUXT8Y`
-   BTC :  `3Jg45PbpUNtASL5uvQSeJKZyiExqbB9mbC`

#### Suggestions/Issues ?
Open a Issue or Write to me : [faceid@pranavms.ml](mailto:faceid@pranavms.ml)

## PR's are welcome.
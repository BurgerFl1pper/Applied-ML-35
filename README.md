# Applied Machine Learning

*Welcome to our project!* 
We have build a machine learning model that is trained to predict what rock subgenre a song is based on what the lyrics are.

## Prerequisites
Make sure you have the following software and tools installed:

- *PyCharm*: We recommend using PyCharm as your IDE, since it offers a highly tailored experience for Python development. You can get a free student license [here](https://www.jetbrains.com/community/education/#students/).

- *Pipenv*: Pipenv is used for dependency management. This tools enables users to easily create and manage virtual environments. To install Pipenv, use the following command:
    bash
    $ pip install --user pipenv
    
    For detailed installation instructions, [click here](https://pipenv.pypa.io/en/latest/installation.html).

- *Git LFS*: Instead of committing large files to your repository, you should store and manage them using Git LFS. For installation information, [click here](https://github.com/git-lfs/git-lfs?utm_source=gitlfs_site&utm_medium=installation_link&utm_campaign=gitlfs#installing).

## Git LFS
Due to the large dataset used for training, we reccomend to use Git LFS.

1. Set it up for your user account (only once, not each time you want to use it).
    bash
    git lfs install
    
2. Select the files that Git LFS should manage. To track all files of a certain type, you can use a wildcard as in the command below.
    bash
   git lfs track "*.psd"
    
3. Add .gitattributes to the staging area.
    bash
    git add .gitattributes
    
That's all, you can commit and push as always. The tracked files will be automatically stored with Git LFS.

## Installing dependencies.
To install the dependencies run the following command:
pip install -r requirements.txt

## Running API
To run the app you can run the following command: 
uvicorn API_SVM:app --reload

Once the app is running, you will receive a link for the application, follow that and add "/docs" to the end of it.

## Using the API
Under the heading "\predict", you are able to provide inputs, and receive our model's genre predictions. Copy your lyrics into the Request body code box, replacing the "string" input. Make sure to submit your text in quotation marks. If there are no errors, then you should be able to see the predicted Genres in the "Response" section underneath. Else, the fitting error will appear underneath that.

*Good luck and have fun! 🚀*

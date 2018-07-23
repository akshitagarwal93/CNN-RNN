# Objectives

The learning objectives of this assignment are to:
1.  Learn the Keras APIs for convolutional and recurrent neural networks.
2.  Explore the space of hyper-parameters for convolutional and recurrent
    networks.

# Setup

You will need to set up an appropriate coding environment on whatever computer
you expect to use for this assignment.
Minimally, you should install:

* [git](https://git-scm.com/downloads)
* [Python (version 3.6 or higher)](https://www.python.org/downloads/)
* [numpy (version 1.11 or higher)](http://www.numpy.org/)
* [Keras (version 2.1 or higher)](https://keras.io/)
* [h5py (version 2.7 or higher)](https://www.h5py.org/)
* [pytest](https://docs.pytest.org/)

# Code

You should first clone the repository that GitHub Classroom created for you:
```
git clone git@github.com:UA-ISTA-457-SP18/convolutional-and-recurrent-networks-<your-username>.git
```
where `<your-username>` should be replaced with your GitHub username.
Once you have successfully cloned the repository, you can begin working on the
assignment.

You will need to write the bodies of 4 methods for this assignment.
Read the docstrings for each of these methods carefully; they give detailed
instructions on how you should implement each method.

# Test

Tests have been provided for you in the `test_nn.py` file.
The tests show how each of the methods is expected to be used.
To run all the provided tests, run the ``pytest`` script from the directory
containing ``test_nn.py``.
Initially, you will see output like:
```
============================= test session starts ==============================
platform darwin -- Python 3.6.4, pytest-3.4.2, py-1.5.2, pluggy-0.6.0
rootdir: /Users/bethard/Code/ista457/convolutional-and-recurrent-networks, inifile:
collected 4 items

test_nn.py FFFF                                                          [100%]

=================================== FAILURES ===================================
_________________________________ test_toy_rnn _________________________________

...
>       model, kwargs = nn.create_toy_rnn(input_shape, n_outputs)
E       TypeError: 'NoneType' object is not iterable

test_nn.py:47: TypeError
...
=========================== 4 failed in 3.11 seconds ===========================
```
This indicates that all tests are failing, which is expected since you have not
yet written the code for any of the methods.
Once you have written the code for all methods, you should instead see
something like:
```
============================= test session starts ==============================
platform darwin -- Python 3.6.4, pytest-3.4.2, py-1.5.2, pluggy-0.6.0
rootdir: /Users/bethard/Code/ista457/convolutional-and-recurrent-networks, inifile:
collected 4 items

test_nn.py 1.4 RMSE for RNN on toy problem
.92.0% accuracy for CNN on MNIST sample
.88.9% accuracy for RNN on Youtube comments
.83.8% accuracy for CNN on Youtube comments
.                                                          [100%]

========================== 4 passed in 44.55 seconds ===========================
```

# Submission

As you are working on the code, you should regularly `git commit` to save your
current changes locally.
You should also regularly `git push` to push all saved changes to the remote
repository on GitHub.

To submit your assignment, simply make sure that you have `git push`ed all of
your changes.
Your instructor will then be able to see, run, and test your code.

You should make a habit of checking the GitHub page for your repository to make
sure your changes have been correctly pushed there.
You may also want to check the "commits" page of your repository on GitHub:
there should be a green check mark beside your last commit, showing that your
code passes all of the given tests.
If there is a red X instead, your code is still failing some of the tests.

# Grading

Assignments will be graded primarily on their ability to pass the tests that
have been provided to you.
Assignments that pass all tests will receive at least 80% of the possible
points.
To get the remaining 20% of the points, make sure that your code is using
appropriate data structures, code duplication is minimized, variables have
meaningful names, complex pieces of code are well documented, etc.

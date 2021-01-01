# machine_learning_01

## Notes from learning Machine Learning topics.

* QwikLabs quest: [Perform Foundational Data, ML, and AI Tasks in Google Cloud](https://google.qwiklabs.com/quests/117)
* QwikLabs focus: [AI Platform](https://google.qwiklabs.com/focuses/581)

**Conclusion**: Some background in Machine Learning is needed before doing the rest of the QwikLabs quest.

* Udacity course: https://classroom.udacity.com/courses/ud187
* Completed *Intro to TensorFlow*

## System Requirements

* Python 3.7
* Pip 20.1+
* Poetry 1.1.4+:
  * `pip install poetry`
* (*On Windows 7 or later*) [Microsoft Visual C++ Redistributable for Visual Studio 2015, 2017 and 2019](https://support.microsoft.com/help/2977003/the-latest-supported-visual-c-downloads)
  * Required by tensorflow.  [Details](https://www.tensorflow.org/install/pip#system-requirements).

To setup the virtual environment, run:
  > `poetry install`

## Results

The current polynomial model does not accurately predict the exponential results of the actual equation.  For reference:

```
Actual = $9,375.00, Predicted = $2,305.64
Actual = $93,750.00, Predicted = $2,305.64
Actual = $11,250.00, Predicted = $2,763.28
Actual = $112,500.00, Predicted = $2,763.29
Actual = $15,000.00, Predicted = $3,678.56
Actual = $150,000.00, Predicted = $3,678.58
Actual = $67,500.00, Predicted = $16,492.53
Actual = $675,000.00, Predicted = $16,492.44
```

This problem will be valuable to re-visit later, when I know how to configure the model and layers for an exponential fit.
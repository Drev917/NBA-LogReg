# NBA-Logistic Regression
Analysis on an NBA player dataset to determine:
- How to define a good rebounder, `Good Rebounder`, (> 8 total rebounds per game).
    - This index was categorical, so I converted it to a numerical column called `RebounderNumeric` in order to make a prediction on a binary value.
- What variables in the dataset can be used to make a prediction of whether a player will be a good rebounder.
- How accurate a prediction can be, and how to tune the model to avoid overfitting.

In order to test the model, I created 3 players with hypothetical statistics based upon the chosen predictor variables:
- `Pos` - Position
- `MP` - Minutes Played
- `PS/G` - Points Scored Per Game
- `AST` - Assists Per Game
- `STL` - Total Steals Per Game

I stacked these parameters vertically using Numpy into a new test set and ran a prediction on `RebounderNumeric`.

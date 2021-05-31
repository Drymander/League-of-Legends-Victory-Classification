
  
  
 # League of Legends - Victory Classification Based on First 10 Minutes

**Author**: [Johnny Dryman](mailto:johnnydryman@gmail.com)

## Overview

**League of Legends** is a team-based strategy game where two teams of five powerful champions face off to destroy the other's base.  This project will use machine learning to predict the winner of a match using only the first 10 minutes of a ~35 minute game.  

## Business Problem

League of Legends (LoL) is an intensely competitive game designed for 'core' gamers, or gamers who care deeply about winning the game. Naturally, these players hate losing. When I first started playing, I noticed that one of my teammates might "rage quit" a game within the first 10-20 minutes, leaving my teammates and I to an inevitable failure. This made my experience suffer, and I could see my teammates lamenting in the chat as well. Considering a single game of LoL can last for 45 minutes, the remaining 35 minutes was not an enjoyable experience.

When I discovered the dataset for League of Legends matches with only the first 10 minutes of data, I was inspired to try and find out whether or not my angry teammates were justified in rage quitting. Was too much of the 45 minute game decided within the first 10 minutes? Can I use machine learning to predict the winner with only 10 minutes of data? What factors in the early game were most likely to lead to a victory?

The goal of this project is to take a look at how well a winner can be predicted based on early game information, and it also seeks to understand what specific factors can predict a victory.

## Data


This data was sourced from Kaggle, where a user posted the first 10 minutes of data pulled from the League of Legends API.  The data represents 9,879 games from Diamond ranked players.  Each row in our dataset represents a single match.  

Here is a brief description of each feature:

 - **blueWins** - a binary column indicating a victory for blue.  0 is a loss, 1 is a victory.
 - **blueWardsPlaced** - the amount of wards blue has placed.  Wards are items that players can place on the map to reveal nearby areas.
 - **blueWardsDestroyed** - the amount of blue wards that have been destroyed (usually by the red team)
 - **blueFirstBlood** - 'first blood' is awarded to the player that gets the first kill
 - **blueKills** - number of the total times a blue player killed a red player
 - **blueDeaths** - number of the total times a red player has killed a blue player
 - **blueAssists** - number of assists, where a blue player assists another blue player in killing a red player
 - **blueEliteMonsters** - number of elite monsters blue has eliminated
 - **blueDragons** - number of times blue has eliminated the dragon (counts towards blueEliteMonsters)
 - **blueHeralds** - number of times blue has eliminated the herald (counts towards blueEliteMonsters)
 - **blueTowersDestroyed** - number of enemy defense towers blue has destroyed
 - **blueTotalGold** - total gold count for blue team among all players, gold is awarded for performing various actions
 - **blueAvgLevel** - average level of all blue players, players start at level 1 at the beginning of the match and maxes out at level 18
 - **blueTotalExperience** - total experience count for all blue players, experience is awarded for performing various actions
 - **blueTotalMinionsKilled** - number of enemy minions (non player characters) blue has killed
 - **blueTotalJungleMinionsKilled** - number of neutral minions (belonging to no team) blue has killed
 - **blueGoldDiff** - the amount of gold blue has over red
 - **blueExperienceDiff** - the amount of experience blue has over red
 - **blueCSPerMin** - number of minions, monsters, and enemy player pets blue has eliminated per minute
 - **blueGoldPerMin** - amount of gold blue team has earned on average every minute

The dataset also features the same information for the red team, (e.g. blueTotalGold, blueTotalExperience, etc.).

## Exploratory Analysis

Let's take a brief look at our data before running any models.

The win split between blue and red seems pretty even.  Of the 9,879 games, blue won 4,930 and red won 4,949

![win_counts](./images/win_counts.png)

Taking a look at blue vs red kills in determining a victory, it's evident with every increase in the kill spread, blue has a higher probability of winning the match.

![kill_spread](./images/kill_spread.PNG)

Killing the dragon in the first 10 minutes seems to have a positive influence on winning. While it might be telling of a win, there are plenty of instances where blue kills the dragon, but red kills the game.  In instances where blue defeats the dragon, their win rate is roughly 64%.  When no team defeats the dragon, our dataset shows that each team's win rate is roughly 50%.

![dragon](./images/dragon.PNG)

In terms of multicollinearity, there is high correlation between blueKills and blueTotalGold as well as blueAvgLvl and blueTotalExperience.

![multicollinearity](./images/multicollinearity.PNG)

## Machine Learning Models

Now that we have established some understanding of the data, let's take a look at how effectively various machine learning models can predict the outcome of a match.

### Logistic Regression with Grid Search CV

Logistic regression with grid search CV was the best performing model with a test accuracy of 72.10%.  Perhaps simplicity works best with this dataset.  

#### Performance

    Training Accuracy: 74.16%
    Test Accuracy: 72.10%

              precision    recall  f1-score   support

           0       0.72      0.72      0.72      1493
           1       0.72      0.72      0.72      1471

    accuracy                           0.72      2964
    macro avg      0.72      0.72      0.72      2964
    weighted avg   0.72      0.72      0.72      2964

![logistic](./images/logistic.PNG)

#### Feature Importance with Coefficients

By plotting the coefficients, we can get a view of what our logistic regression model deems most important to predicting victory.

![coeff_all](./images/coeff_all.PNG)

Total gold is by far the most important feature, while eliminating the dragon and total experience are secondarily important.  Surprisingly, blue kills has almost no importance in this model, which is counterintuitive to what seasoned League of Legends players might expect.

Taking it one step further, we can rerun the model and remove total gold, total experience, and average level.  Not only were these features deemed highly correlated by our multicollinearity analysis, they also represent rewards for performing in-game actions and are heavily tied to with a number of other features.  Removing them and running a new logistic regression would highlight what actions are most important in the game.

![coeff_action](./images/coeff_action.PNG)

This paints a much clearer picture of what actions are most predictive of a win.  Importantly, rerunning the logistic regression with these features produced an accuracy of 70.68%, or only 1.42% less predictive than our best performing model.

Now we can see that kills are most important win predicting victory.  Eliminating the dragon and minions are also important.

### Random Forest with Grid Search CV

Let's try a more powerful model.  Random forests consist of a large number of individual decision trees that act together for classification.  Using grid search CV, we can run hundreds or thousands of random forests with a combination of parameters, and the grid search will return the parameters with the best performance.  

#### Performance

Our random forest with grid search did not outperform our logistic regression.

    Training Accuracy: 76.25%
    Test Accuracy: 71.49%

              precision    recall  f1-score   support

           0       0.72      0.71      0.72      1493
           1       0.71      0.72      0.71      1471

    accuracy                           0.71      2964
    macro avg      0.71      0.71      0.71      2964
    weighted avg   0.71      0.71      0.71      2964

![randomforest](./images/randomforest.PNG)

#### Random Forest Feature Importance

Random forests will also let us view feature importances, but these do not represent coefficients like in our logistic regression features.  

Where logistic regression had some counterintuitive choices for feature importance, random forest is much more in line with our expectations.  Total gold, total experience, and average level, which collectively represent a more precise score for how each team is performing, are most highly valued.  In terms of actions, dragon is ranked higher than kills, but they are still similarly important.


![rf_features](./images/rf_features.PNG)

### XGBoost with Grid Search CV

Finally, we will try XGBoost to see if we can outperform our logistic regression.  XGBoost works similarly to random forest, but uses "boosting," which is based high bias, low variance learners.  Boosting reduces bias by adding one classifier at a time so that each consequent classifier is trained to improve the already trained ensemble.  

#### Performance

While XGBoost with grid search CV outperformed our random forest model by 0.25%, it failed to provide any more predictive power than our logistic regression with grid search CV.  

    Training Accuracy: 75.84%
    Test Accuracy: 71.73%

              precision    recall  f1-score   support

           0       0.72      0.72      0.72      1493
           1       0.71      0.72      0.72      1471

    accuracy                           0.72      2964
    macro avg      0.72      0.72      0.72      2964
    weighted avg   0.72      0.72      0.72      2964

![xgboost](./images/xgboost.PNG)


#### XGBoost Feature Importance

XGBoost's feature importances are very similar to those from our random forest model, which again falls in line with what we might normally expect based on our knowledge of League of Legends.

![xgb_features](./images/xgb_features.PNG)

# Interpretation

While somewhat disappointing that we couldn't get any higher than 72.10% accuracy with any of our models, this realization is ultimately good news for the League of Legends team. If higher accuracy could be achieved, it would indicate that the outcome of the game might rely too heavily on the first 10 minutes. That said, it's worth considering whether or not 72% is too high.

Our best performing model was logistic regression with grid search. An important observation from our random forest and XGBoost feature importances were that red and blue differences were somewhat uneven, which begs to question whether or not these models are appropriate for this type of dataset.

All of the models seemed to point to indicators of early success rather than the actions players can take. In all models, TotalGold, TotalExperience, and AvgLevel took very high if not highest feature importance. Interestingly, when we removed these features from our dataset, accuracy did not suffer more than 2-3%.

In terms of what actions are most valuable, TotalKills, TotalMinionsKilled, TotalJungleMinionsKilled, and defeating the dragon are most important.

# Conclusions and Recommendations

The goal of this project was to determine whether or not Riot should consider balancing the game based on our findings. Ultimately, we could not find any glaring issues with overweighted feature importance. The fact that none of our machine learning models could best 72% accuracy in predicting the winner is a sign that the first 10 minutes is not overtly important, but is 72% too high?

Would something like 60-65% make for a more engaging player experience and encourage players to finish strong throughout the match?

It's also important to understand that this data was pulled from the most skilled players in League of Legends online matchmaking. LoL's most core audience would not likely react too positively to strong shifts catering towards newer players.

We propose the following questions to League of Legends developers for consideration:

-   Is 72% predictive quality too high for the first 10 minutes of League of Legends?
-   Are kills too heavily weighted in terms of actions that players take?
-   Could tinkering with buffs granted by dragon and herald lead to changes in predictive quality?
-   Are gold rewards too high in the early game?
-   Most importantly, what amount of predictive quality is desirable for the best player experience, both for those winning and those losing?

# Future Analysis

For future analysis, we would recommend running this dataset with the exact same matches from this dataset but with 20 minutes and 30 minutes of data to see how predictive quality changes. Would predictive quality improve, or would it stay the same? Also, more elements would be introduced at that point, and those additional features should be taken into consideration as well.

We would also ask the League of Legends staff where the true issues are with their current player base. Are the highly skilled players content, but newer players less so? Is new player retention equal, higher, or lower priority than keeping their dedicated fans happy? There is likely no best answer to this question. Tinkering with games that have a loyal fanbase is a delicate and sometimes detrimental act, and it would be helpful to understand the history of game modifications and player approval before recommending anything formal.

## For More Information

See the full analysis in the [Jupyter Notebook](./Johnny Dryman - Phase 3 Project Notebook.ipynb) or review this [presentation](./Johnny Dryman - Phase 3 Project Presentation.pdf).

For additional info, contact Johnny Dryman at [johnnydryman@gmail.com](mailto:johnnydryman@gmail.com)

## Repository Structure

```
├── data
├── images
├── README.md
├── Johnny Dryman - Phase 3 Project Presentation.pdf
└── Johnny Dryman - Phase 3 Project Notebook.ipynb
```
# Inversion Forecast

This proyect yas quite challenging because I had to predict the investment of the competitors of 4 South American countries. The problema here was that I only had 10 years of history and they wanted to predict 2 years ahead! So I decided to use models with high bias and trait data in a very carefully way. I land on an R2 of 0.7... kind of acceptable for this problem.

My input data was conposed by:

- Nielsen data (sales by month)
- IBOPE data (investment opened by channels and ratio on that media)
- Macroeconomic data (inflation, demos, PBI, etc)
- Weather data (without sense.. because it was investment not sales but I forced to added it)

#### Final Pipeline:

![Screenshot](Pipeline.png)


#### Some comments:

- competenciaVSkoAnual2.py Forcast the potencial investment for the punctual brand, opened by category. The model was done in two steps because the amount of history wasn't enough, so I predict 2018 and then 2019 whith 2018 output.
- competenciaVSkoAnualGRPS.py Forcast the potencial TRPs for the punctual brand, opened by category. The model need the forecasted investment of 2018 and 2019. This is because CPR (cost per raiting) must move in a coherent way. 



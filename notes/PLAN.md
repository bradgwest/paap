# Stat 575 Plan

## Obtaining Data
* Christie's and Sotheby's are the two main auction houses for fine art. Their websites contain historical auction results from at least the past decade, a corpus of tens of thousands of paintings.
* Scrape auction results from these websites, returning all prints, drawings, and paintings (Including sculptures, antiques, etc., is our of the scope.
* The result of scraping should be a dataset of prints, drawings, and paintings with n > 50,000, all of which have an associated image, price data, and artist data. Specifically, the feature set at this point will contain:
  + Sale price
  + Sale currency
  + Pre-sale estimate (as provided by the auction house)
  + Number of bids
  + Auction date
  + Auction location
  + Artist name
  + Artist date of birth
  + Artist date of death
  + Artist nationality
  + Piece date (if known)
  + Piece medium
  + Piece dimensions
  + Piece title

## Feature Engineering
### Images
* From the image data we also want to extract some basic features:
  + Brightness
  + Smoothness
  + Color Pallete -- Via clustering
* There are a number of other features we can extract, but I would like to start with these

### Artist Data
* We expect some artists works to sell for more due to the artist. It may be useful to get a measure of artist significance. Past researchers have used a combination of Google page rank and Wikipedia to get significance. Time allowing it would interesting to research a method for getting this data and incorporating it into our feature set.


## Research Questions
The following seem like interesting questions:`
  1. What variables are most useful in predicting art prices?
  2. How much do image characteristics play in auction prices?
  3. After controlling for artist fame, what controls prices?
  4. Do the auction houses differ in some of these features?
  5. 

## EDA
* Correlation between predictors
* Is this a linear or highly non-linear feature set?
* I assume we will need to control for right tail skew

## Models
* Linear regression
* Nonlinear
  + KNN, SVM, Regression Trees, Neural Nets

# TODO
* Share github with Andy

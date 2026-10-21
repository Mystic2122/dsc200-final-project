# DSC 200 Project Proposal

**DSC 200: Data Science Programming — Fall 2026**

## Group

- **Student names:** Kyra Deng, Melody Fang, Vance Liu, Nicholas Wendt
- **Contact person:** Melody Fang
- **Proposed project title:** Personalized Restaurant Recommendation Using Yelp Data

## Problem and Intended User

- The problem we'd like to address is indecision, decision fatigue, and information overload when it comes to picking a restaurant based on personal preferences (e.g. cost, location, parking, etc.).
- Our project is intended to be used by anyone who needs help picking a place to eat or wants recommendations.

## Data

We will use data from Yelp's [public dataset](https://business.yelp.com/data/resources/open-dataset/) containing six files, two of which are of interest (review information and business information). The data is distributed as JSON files, consisting of ~7 million separate reviews and ~150,000 businesses, 4.35 GB compressed and 8.65 GB uncompressed. We will mostly use the business and review files, filtered down to restaurants only. The dataset is public and free to download after agreeing to Yelp's Dataset Terms of Use, which permit educational, non-commercial use. We have downloaded the data and will not redistribute it.

## Definition of "Done"

- The primary deliverable is being able to return restaurant recommendations, chosen and ranked by a weighted score based on user-set preferences.
- Recommendations should be filterable based on how many recommendations are desired, parking, open times, etc.
- We plan to create a very basic user interface where preferences can be set and recommendations can be displayed.
- Other features we are considering include image classification of the pictures included in the reviews and what information they provide, as well as possibly expanding the tool to allow users and friends to input preferences together for a balanced recommendation.

## Questions or Concerns

Our concerns include:

- The dataset is very large (for reviews), which can be difficult to store and to access. We will filter down to only restaurant-related data, and would appreciate guidance on how to optimize.
- The dataset currently includes some nested objects in the attributes, which can lead to issues with pandas processing. We will consider incorporating parsing first.
- Without past reviews or other data on a new user, it may be hard to learn their preferences and make relevant recommendations.
- Yelp's provided dataset with real-world data allows for free educational and non-commercial use. We will follow the terms for using the data, including no redistribution, and do not anticipate any licensing issues within the scope of this project.
- The dataset only covers 11 metropolitan areas, which may be a concern for the location part of recommending restaurants.

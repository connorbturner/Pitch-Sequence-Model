### Data Scraper for Conditional Pitch Sequence Prediction Model
### Author: Connor Turner

# Load Necessary Packages -------------------------------------------------

library(baseballr)
library(tidyverse)


# Gather Pitch Data for Each Year -----------------------------------------

# Because there is so much data from each season, the scraping has to be
# done in small batches of games in order to avoid issues:

### Set initial date

year = 2018
month = 3
day = 29
start = paste0(year, "-", month, "-", day)
end = paste0(year, "-", month, "-", day)

### Run first pass

pitch_data <- statcast_search(start, end)


### Gather rest of data for the year

repeat {
  day = day + 1
  start = paste0(year, "-", month, "-", day)
  end = paste0(year, "-", month, "-", day)
  
  if (day == 30){
    month = month + 1
    day = 1
    end = paste0(year, "-", month, "-", day)
  }
  
  if (start == "2018-10-3"){
    break
  }
  
  new_search <- statcast_search(start, end)
  pitch_data <- rbind(pitch_data, new_search)
}


# Match Player Names with IDs ---------------------------------------------

# For identification and presentation purposes, we need to attach the
# names for the pitcher and hitter using the provided MLBAM IDs in the data

### Gather and store unique IDs

id <- unique(c(pitch_data$batter, pitch_data$pitcher))
player_ids <- data_frame(id)
player_ids$name <- NA


### Find the names associated with each ID

for (i in 1:nrow(player_ids)){
  lookup <- playername_lookup(id[i])
  new_name <- paste(lookup[[1]], lookup[[2]])
  player_ids$name[i] = new_name
}


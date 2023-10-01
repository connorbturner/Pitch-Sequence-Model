### Data Builder for Conditional Pitch Sequence Prediction Model
### Author: Connor Turner

# Load Necessary Packages -------------------------------------------------

library(baseballr)
library(tidyverse)


# Load Datasets -----------------------------------------------------------

# Regular-season pitch-by-pitch data from 2018-2023 (2020 excluded)
pitch_data = read.csv("pitch_data.csv")

# Player names and associated MLBAM IDs
player_ids <- read.csv("player_ids.csv") %>% select(c(id, name))


# Build dataset for model -------------------------------------------------

model_data <- pitch_data %>% 
  
  ### Select relevant columns
  
  select(c(game_date, game_year, game_pk, at_bat_number, pitch_number, batter, 
           pitcher, stand, p_throws, balls, strikes, outs_when_up, pitch_type, 
           pitch_name, type, events, description, on_1b, on_2b, on_3b)) %>% 
  
  
  ### Add batter and pitcher names
  
  rename(id = batter) %>% 
  left_join(player_ids, by = "id") %>% 
  rename(batter_id = id,
         batter_name = name,
         id = pitcher) %>% 
  left_join(player_ids, by = "id") %>% 
  rename(pitcher_id = id,
         pitcher_name = name) %>% 
  
  
  ### Arrange the data by date, game, at bat, and pitch
  
  arrange(game_date, game_pk, at_bat_number, pitch_number) %>% 
  
  
  ### Filter out missing data, pitch outs, and rare pitches
  
  group_by(game_pk, at_bat_number) %>% 
  filter(!any(pitch_name == "" | pitch_name == "Pitch Out" | 
                pitch_name == "Eephus" | pitch_name == "Knuckleball" |
                pitch_name == "Other")) %>% 
  ungroup() %>% 
  
  ### Filter out batters with less than 1000 pitches in a given year
  
  group_by(batter_id, game_year) %>% 
  filter(n() >= 1000) %>% 
  ungroup() %>% 
  
  
  ### Filter out pitchers with less than 1000 pitches in a given year
  
  group_by(pitcher_id, game_year) %>% 
  filter(n() >= 1000) %>% 
  ungroup() %>% 
  
  
  ### Consolidate pitch types into categories
  
  # 1) Fastball = "4-Seam Fastball"
  # 2) Sinker = "Sinker"
  # 3) Cutter = "Cutter"
  # 4) Slider = "Slider", "Sweeper"
  # 5) Curveball = "Curveball", "Knuckle Curve", "Slurve", "Slow Curve"
  # 6) Changeup = "Changeup", "Screwball"
  # 7) Splitter = "Split-Finger", "Forkball"

  mutate(# Record the pitch type that was thrown
         pitch = ifelse(pitch_name == "4-Seam Fastball", 1, 0),
         pitch = ifelse(pitch_name == "Sinker", 2, pitch), 
         pitch = ifelse(pitch_name == "Cutter", 3, pitch), 
         pitch = ifelse(pitch_name %in% c("Slider", "Sweeper"), 4, pitch), 
         pitch = ifelse(pitch_name %in% c("Curveball", "Knuckle Curve", 
                                           "Slurve", "Slow Curve"), 5, pitch), 
         pitch = ifelse(pitch_name %in% c("Changeup", "Screwball"), 6, pitch), 
         pitch = ifelse(pitch_name %in% c("Split-Finger","Forkball"), 7, pitch), 
         
         # Record the 3 previous pitch types
         prev_pitch = ifelse(pitch_number == 1, 0, lag(pitch)), 
         prev_pitch_2 = ifelse(pitch_number <= 2, 0, lag(pitch, 2)), 
         prev_pitch_3 = ifelse(pitch_number <= 3, 0, lag(pitch, 3))) %>% 
  
  
  ### Calculate general probabilities for pitchers
  
  group_by(pitcher_id, game_year, stand) %>% 
  mutate(g_p1 = sum(pitch == 1) / n(),
         g_p2 = sum(pitch == 2) / n(),
         g_p3 = sum(pitch == 3) / n(),
         g_p4 = sum(pitch == 4) / n(),
         g_p5 = sum(pitch == 5) / n(),
         g_p6 = sum(pitch == 6) / n(),
         g_p7 = sum(pitch == 7) / n()) %>% 
  ungroup() %>% 
  
  
  ### Calculate conditional probabilities for pitchers
  
  group_by(pitcher_id, game_year, balls, strikes, prev_pitch, stand) %>% 
  mutate(p_p1 = sum(pitch == 1) / n(),
         p_p2 = sum(pitch == 2) / n(),
         p_p3 = sum(pitch == 3) / n(),
         p_p4 = sum(pitch == 4) / n(),
         p_p5 = sum(pitch == 5) / n(),
         p_p6 = sum(pitch == 6) / n(),
         p_p7 = sum(pitch == 7) / n()) %>% 
  ungroup() %>% 
  
  
  ### Calculate conditional probabilities for hitters
  
  group_by(batter_id, game_year, balls, strikes, prev_pitch, p_throws) %>% 
  mutate(b_p1 = sum(pitch == 1) / n(),
         b_p2 = sum(pitch == 2) / n(),
         b_p3 = sum(pitch == 3) / n(),
         b_p4 = sum(pitch == 4) / n(),
         b_p5 = sum(pitch == 5) / n(),
         b_p6 = sum(pitch == 6) / n(),
         b_p7 = sum(pitch == 7) / n()) %>% 
  ungroup() %>% 
  
  ### Create new variables with relevant information
  
  mutate(# Dummy for previous pitch type
         pp1_1 = ifelse(prev_pitch == 1, 1, 0),
         pp1_2 = ifelse(prev_pitch == 2, 1, 0), 
         pp1_3 = ifelse(prev_pitch == 3, 1, 0), 
         pp1_4 = ifelse(prev_pitch == 4, 1, 0), 
         pp1_5 = ifelse(prev_pitch == 5, 1, 0), 
         pp1_6 = ifelse(prev_pitch == 6, 1, 0), 
         pp1_7 = ifelse(prev_pitch == 7, 1, 0), 
         
         # Dummy for second most recent pitch type
         pp2_1 = ifelse(prev_pitch_2 == 1, 1, 0), 
         pp2_2 = ifelse(prev_pitch_2 == 2, 1, 0), 
         pp2_3 = ifelse(prev_pitch_2 == 3, 1, 0), 
         pp2_4 = ifelse(prev_pitch_2 == 4, 1, 0), 
         pp2_5 = ifelse(prev_pitch_2 == 5, 1, 0), 
         pp2_6 = ifelse(prev_pitch_2 == 6, 1, 0), 
         pp2_7 = ifelse(prev_pitch_2 == 7, 1, 0), 
         
         # Dummy for third most recent pitch type
         pp3_1 = ifelse(prev_pitch_3 == 1, 1, 0), 
         pp3_2 = ifelse(prev_pitch_3 == 2, 1, 0), 
         pp3_3 = ifelse(prev_pitch_3 == 3, 1, 0), 
         pp3_4 = ifelse(prev_pitch_3 == 4, 1, 0), 
         pp3_5 = ifelse(prev_pitch_3 == 5, 1, 0), 
         pp3_6 = ifelse(prev_pitch_3 == 6, 1, 0), 
         pp3_7 = ifelse(prev_pitch_3 == 7, 1, 0), 
         
         # Dummy variables denoting if runners are on first, second, or third
         man_on_1b = ifelse(!is.na(on_1b), 1, 0),
         man_on_2b = ifelse(!is.na(on_2b), 1, 0), 
         man_on_3b = ifelse(!is.na(on_3b), 1, 0),
         
         # Dummy denoting the number of outs
         out_0 = ifelse(outs_when_up == 0, 1, 0),
         out_1 = ifelse(outs_when_up == 1, 1, 0),
         out_2 = ifelse(outs_when_up == 2, 1, 0),
         
         # Dummy denoting what pitch is thrown next (target vector)
         p_1 = ifelse(pitch == 1, 1, 0), 
         p_2 = ifelse(pitch == 2, 1, 0), 
         p_3 = ifelse(pitch == 3, 1, 0), 
         p_4 = ifelse(pitch == 4, 1, 0), 
         p_5 = ifelse(pitch == 5, 1, 0), 
         p_6 = ifelse(pitch == 6, 1, 0), 
         p_7 = ifelse(pitch == 7, 1, 0)) %>% 
  
  ### Build the final dataset
  
  select(# Pitcher and batter information
         batter_id, batter_name, pitcher_id, pitcher_name,
         # Situational information
         stand, p_throws, balls, strikes, 
         # Pitcher general probabilities
         g_p1, g_p2, g_p3, g_p4, g_p5, g_p6, g_p7,
         # Pitcher conditional probabilities
         p_p1, p_p2, p_p3, p_p4, p_p5, p_p6, p_p7,
         # Batter conditional probabilities
         b_p1, b_p2, b_p3, b_p4, b_p5, b_p6, b_p7,
         # Previous pitch dummy
         pp1_1, pp1_2, pp1_3, pp1_4, pp1_5, pp1_6, pp1_7, 
         # Second-most recent pitch dummy
         pp2_1, pp2_2, pp2_3, pp2_4, pp2_5, pp2_6, pp2_7, 
         # Third-most recent pitch dummy
         pp3_1, pp3_2, pp3_3, pp3_4, pp3_5, pp3_6, pp3_7, 
         # Base occupation dummy
         man_on_1b, man_on_2b, man_on_3b,
         # Out dummy
         out_0, out_1, out_2,
         # Pitch dummy / target vector
         p_1, p_2, p_3, p_4, p_5, p_6, p_7)               


# Split Data and Export ---------------------------------------------------

### Split data into training, validation, and testing sets

n <- nrow(model_data)                           # Number of observations
set.seed(12345)                                 # Set seed for reproducibility

id_tr <- sample(1:n, floor(n*0.5))              # Set training data indices
model_training_data <- model_data[id_tr,]       # Assign training data
id_1 <-  setdiff(1:n, id_tr)                    # Find remaining indices

set.seed(12345)
id_val <- sample(id_1, floor(n*0.3))            # Select remaining indices
model_validation_data <- model_data[id_val,]    # Assign validation data
id_te <- setdiff(id_1,id_val)                   # Return remaining indices
model_testing_data <- model_data[id_te,]        # Assign testing data


### Build smaller versions of the data for testing purposes

small_training_data <- model_training_data[1:50000,]
small_validation_data <- model_validation_data[1:30000,]
small_testing_data <- model_testing_data[1:20000,]


### Export the data sets

write.csv(model_data, "psm_model_data.csv")
write.csv(model_training_data, "psm_training_data.csv")
write.csv(model_validation_data, "psm_validation_data.csv")
write.csv(model_testing_data, "psm_testing_data.csv")
write.csv(small_training_data, "psm_training_data_small.csv")
write.csv(small_validation_data, "psm_validation_data_small.csv")
write.csv(small_testing_data, "psm_testing_data_small.csv")


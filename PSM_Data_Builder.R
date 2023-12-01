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


# Build Dataset for Model -------------------------------------------------

count_map <- c("0-0" = 1L, "0-1" = 2L, "1-0" = 3L, "1-1" = 4L, "0-2" = 5L,
               "2-0" = 6L, "1-2" = 7L, "2-1" = 8L, "3-0" = 9L, "3-1" = 10L, 
               "2-2" = 11L, "3-2" = 12L)

data <- pitch_data %>% 
  
  ### Select relevant columns:
  
  select(c(game_date, game_year, game_pk, inning, at_bat_number, pitch_number, 
           batter, pitcher, stand, p_throws, balls, strikes, outs_when_up, 
           pitch_type, pitch_name, zone, type, events, on_1b, on_2b, on_3b)) %>% 
  
  
  ### Add batter and pitcher names:
  
  rename(id = batter) %>% 
  left_join(player_ids, by = "id") %>% 
  rename(batter_id = id,
         batter_name = name,
         id = pitcher) %>% 
  left_join(player_ids, by = "id") %>% 
  rename(pitcher_id = id,
         pitcher_name = name) %>% 
  mutate(pitcher_name = paste(game_year, pitcher_name),
         pitcher_id = as.integer(paste0(game_year, pitcher_id))) %>% 
  
  
  ### Arrange the data by date, game, at bat, and pitch:
  
  arrange(game_date, game_pk, at_bat_number, pitch_number) %>% 
  
  
  ### Filter out missing data, outlier pitches, and at-bats that are too long:
  
  group_by(game_pk, at_bat_number) %>% 
  filter(!any(pitch_name == "" | pitch_name == "Pitch Out" | 
                pitch_name == "Eephus" | pitch_name == "Knuckleball" |
                pitch_name == "Other"),
         !any(pitch_number > 8),
         !any(inning > 9),
         !any(balls > 3), 
         !any(strikes > 2),
         !any(is.na(zone))) %>% 
  ungroup() %>% 
  
  
  ### Filter out pitchers with less than 1000 pitches in a given year:
  
  group_by(pitcher_id, game_year) %>% 
  filter(n() >= 1000) %>% 
  ungroup() %>% 
  
  
  ### Consolidate pitch types into categories:
  
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
  
  
  ### Calculate conditional probabilities for pitchers:
  
  group_by(pitcher_id, game_year, balls, strikes, prev_pitch, stand) %>% 
  mutate(p_p1 = sum(pitch == 1) / n(),
         p_p2 = sum(pitch == 2) / n(),
         p_p3 = sum(pitch == 3) / n(),
         p_p4 = sum(pitch == 4) / n(),
         p_p5 = sum(pitch == 5) / n(),
         p_p6 = sum(pitch == 6) / n(),
         p_p7 = sum(pitch == 7) / n()) %>% 
  ungroup() %>% 
  
  
  ### Calculate conditional probabilities for hitters:
  
  group_by(batter_id, game_year, balls, strikes, prev_pitch, p_throws) %>% 
  mutate(b_p1 = sum(pitch == 1) / n(),
         b_p2 = sum(pitch == 2) / n(),
         b_p3 = sum(pitch == 3) / n(),
         b_p4 = sum(pitch == 4) / n(),
         b_p5 = sum(pitch == 5) / n(),
         b_p6 = sum(pitch == 6) / n(),
         b_p7 = sum(pitch == 7) / n()) %>% 
  ungroup() %>% 
  
  
  ### Create new variables with relevant information:
  
  mutate(# Dummies for previous pitch type
         pp1_1 = ifelse(prev_pitch == 1, 1, 0),
         pp1_2 = ifelse(prev_pitch == 2, 1, 0), 
         pp1_3 = ifelse(prev_pitch == 3, 1, 0), 
         pp1_4 = ifelse(prev_pitch == 4, 1, 0), 
         pp1_5 = ifelse(prev_pitch == 5, 1, 0), 
         pp1_6 = ifelse(prev_pitch == 6, 1, 0), 
         pp1_7 = ifelse(prev_pitch == 7, 1, 0),  
         
         pp2_1 = ifelse(prev_pitch_2 == 1, 1, 0),
         pp2_2 = ifelse(prev_pitch_2 == 2, 1, 0), 
         pp2_3 = ifelse(prev_pitch_2 == 3, 1, 0), 
         pp2_4 = ifelse(prev_pitch_2 == 4, 1, 0), 
         pp2_5 = ifelse(prev_pitch_2 == 5, 1, 0), 
         pp2_6 = ifelse(prev_pitch_2 == 6, 1, 0), 
         pp2_7 = ifelse(prev_pitch_2 == 7, 1, 0), 
         
         pp3_1 = ifelse(prev_pitch_3 == 1, 1, 0),
         pp3_2 = ifelse(prev_pitch_3 == 2, 1, 0), 
         pp3_3 = ifelse(prev_pitch_3 == 3, 1, 0), 
         pp3_4 = ifelse(prev_pitch_3 == 4, 1, 0), 
         pp3_5 = ifelse(prev_pitch_3 == 5, 1, 0), 
         pp3_6 = ifelse(prev_pitch_3 == 6, 1, 0), 
         pp3_7 = ifelse(prev_pitch_3 == 7, 1, 0),
         
         # Dummy for pitch location
         zone = as.integer(zone),
         pp_loc = ifelse(pitch_number == 1, 0L, lag(zone)),
         pp2_loc = ifelse(pitch_number > 2, lag(pp_loc), 0L),
         pp3_loc = ifelse(pitch_number > 3, lag(pp2_loc), 0L),
         loc_1 = ifelse(zone == 1, 1, 0), 
         loc_2 = ifelse(zone == 2, 1, 0), 
         loc_3 = ifelse(zone == 3, 1, 0), 
         loc_4 = ifelse(zone == 4, 1, 0), 
         loc_5 = ifelse(zone == 5, 1, 0), 
         loc_6 = ifelse(zone == 6, 1, 0), 
         loc_7 = ifelse(zone == 7, 1, 0), 
         loc_8 = ifelse(zone == 8, 1, 0), 
         loc_9 = ifelse(zone == 9, 1, 0), 
         loc_10 = ifelse(zone == 10, 1, 0), 
         loc_11 = ifelse(zone == 11, 1, 0), 
         loc_12 = ifelse(zone == 12, 1, 0), 
         loc_13 = ifelse(zone == 13, 1, 0), 
         loc_14 = ifelse(zone == 14, 1, 0), 
         
         # Dummy for previous pitch location
         pp_loc_1 = ifelse(pp_loc == 1, 1, 0), 
         pp_loc_2 = ifelse(pp_loc == 2, 1, 0), 
         pp_loc_3 = ifelse(pp_loc == 3, 1, 0), 
         pp_loc_4 = ifelse(pp_loc == 4, 1, 0), 
         pp_loc_5 = ifelse(pp_loc == 5, 1, 0), 
         pp_loc_6 = ifelse(pp_loc == 6, 1, 0), 
         pp_loc_7 = ifelse(pp_loc == 7, 1, 0), 
         pp_loc_8 = ifelse(pp_loc == 8, 1, 0), 
         pp_loc_9 = ifelse(pp_loc == 9, 1, 0), 
         pp_loc_10 = ifelse(pp_loc == 10, 1, 0), 
         pp_loc_11 = ifelse(pp_loc == 11, 1, 0), 
         pp_loc_12 = ifelse(pp_loc == 12, 1, 0), 
         pp_loc_13 = ifelse(pp_loc == 13, 1, 0), 
         pp_loc_14 = ifelse(pp_loc == 14, 1, 0), 
         
         # Pitch number dummies
         pn_1 = ifelse(pitch_number == 1, 1, 0),
         pn_2 = ifelse(pitch_number == 2, 1, 0),
         pn_3 = ifelse(pitch_number == 3, 1, 0),
         pn_4 = ifelse(pitch_number == 4, 1, 0),
         pn_5 = ifelse(pitch_number == 5, 1, 0),
         pn_6 = ifelse(pitch_number == 6, 1, 0),
         pn_7 = ifelse(pitch_number == 7, 1, 0),
         pn_8 = ifelse(pitch_number == 8, 1, 0),
         
         # Inning Dummies
         inn_1 = ifelse(inning == 1, 1, 0),
         inn_2 = ifelse(inning == 2, 1, 0),
         inn_3 = ifelse(inning == 3, 1, 0),
         inn_4 = ifelse(inning == 4, 1, 0),
         inn_5 = ifelse(inning == 5, 1, 0),
         inn_6 = ifelse(inning == 6, 1, 0),
         inn_7 = ifelse(inning == 7, 1, 0),
         inn_8 = ifelse(inning == 8, 1, 0),
         inn_9 = ifelse(inning == 9, 1, 0),
         
         # Count Variables
         count = paste0(balls, "-", strikes),
         count_code = count_map[count],
         count_00 = ifelse(count_code == 1, 1, 0),
         count_01 = ifelse(count_code == 2, 1, 0),
         count_10 = ifelse(count_code == 3, 1, 0),
         count_11 = ifelse(count_code == 4, 1, 0),
         count_02 = ifelse(count_code == 5, 1, 0),
         count_20 = ifelse(count_code == 6, 1, 0),
         count_12 = ifelse(count_code == 7, 1, 0),
         count_21 = ifelse(count_code == 8, 1, 0),
         count_30 = ifelse(count_code == 9, 1, 0),
         count_31 = ifelse(count_code == 10, 1, 0),
         count_22 = ifelse(count_code == 11, 1, 0),
         count_32 = ifelse(count_code == 12, 1, 0),
         
         # Dummy variables denoting if batter is left- or right-handed
         bats_left = ifelse(stand == "L", 1, 0),
         bats_right = ifelse(stand == "R", 1, 0),
         bat_stand = ifelse(stand == "R", 1, 2),
         
         # Dummy variables denoting if runners are on first, second, or third
         b1 = ifelse(!is.na(on_1b), 1, 0),
         b2 = ifelse(!is.na(on_2b), 1, 0), 
         b3 = ifelse(!is.na(on_3b), 1, 0),
         base_sit = 0,
         base_sit = ifelse(b1 == 1 & b2 == 0 & b3 == 0, 1, base_sit),
         base_sit = ifelse(b1 == 0 & b2 == 1 & b3 == 0, 2, base_sit),
         base_sit = ifelse(b1 == 0 & b2 == 0 & b3 == 1, 3, base_sit),
         base_sit = ifelse(b1 == 1 & b2 == 1 & b3 == 0, 4, base_sit),
         base_sit = ifelse(b1 == 1 & b2 == 0 & b3 == 1, 5, base_sit),
         base_sit = ifelse(b1 == 1 & b2 == 1 & b3 == 1, 6, base_sit),
         base_sit = ifelse(b1 == 1 & b2 == 1 & b3 == 1, 7, base_sit),
         
         # Dummy denoting the number of outs
         out_0 = ifelse(outs_when_up == 0, 1, 0),
         out_1 = ifelse(outs_when_up == 1, 1, 0),
         out_2 = ifelse(outs_when_up == 2, 1, 0),
         
         # Dummy denoting the year
         year_18 = ifelse(game_year == 2018, 1, 0), 
         year_19 = ifelse(game_year == 2019, 1, 0),
         year_21 = ifelse(game_year == 2021, 1, 0),
         year_22 = ifelse(game_year == 2022, 1, 0),
         year_23 = ifelse(game_year == 2023, 1, 0),
         year = ifelse(game_year == 2018, 1, 0),
         year = ifelse(game_year == 2019, 2, year),
         year = ifelse(game_year == 2021, 3, year),
         year = ifelse(game_year == 2022, 4, year),
         year = ifelse(game_year == 2023, 5, year),
         
         # Dummy denoting what pitch is thrown next (target vector)
         p_1 = ifelse(pitch == 1, 1, 0), 
         p_2 = ifelse(pitch == 2, 1, 0), 
         p_3 = ifelse(pitch == 3, 1, 0), 
         p_4 = ifelse(pitch == 4, 1, 0), 
         p_5 = ifelse(pitch == 5, 1, 0), 
         p_6 = ifelse(pitch == 6, 1, 0), 
         p_7 = ifelse(pitch == 7, 1, 0))
  

# Build the Final Datasets ------------------------------------------------

### One-hot encoded DNN data:

dnn_ohe_model_data <- data %>% 
  arrange(pitcher_id, game_pk) %>% 
  select(### General Information (7 features):
    
         # Game information
         game_year, game_pk, 
         # Situational information
         pitcher_id, pitcher_name, batter_id, batter_name, count,
    
         ### Feature vector (86 features):
    
         # Pitcher conditional probabilities
         p_p1, p_p2, p_p3, p_p4, p_p5, p_p6, p_p7,
         # Batter conditional probabilities
         b_p1, b_p2, b_p3, b_p4, b_p5, b_p6, b_p7,
         # Pitch number dummy
         pn_1, pn_2, pn_3, pn_4, pn_5, pn_6, pn_7, pn_8,
         # Previous pitch dummy
         pp1_1, pp1_2, pp1_3, pp1_4, pp1_5, pp1_6, pp1_7, 
         # Second-most recent pitch dummy
         pp2_1, pp2_2, pp2_3, pp2_4, pp2_5, pp2_6, pp2_7, 
         # Third-most recent pitch dummy
         pp3_1, pp3_2, pp3_3, pp3_4, pp3_5, pp3_6, pp3_7,
         # Previous pitch location dummy
         pp_loc_1, pp_loc_2, pp_loc_3, pp_loc_4, pp_loc_5, 
         pp_loc_6, pp_loc_7, pp_loc_8, pp_loc_9, pp_loc_10, 
         pp_loc_11, pp_loc_12, pp_loc_13, pp_loc_14, 
         # Count Dummy
         count_00, count_01, count_10, count_11, count_02, count_20, 
         count_12, count_21, count_30, count_31, count_22, count_32,
         # Inning Dummy
         inn_1, inn_2, inn_3, inn_4, inn_5, inn_6, inn_7, inn_8, inn_9,
         # Batter handedness dummy
         bats_left, bats_right,
         # Base occupation dummy
         b1, b2, b3,
         # Out dummy
         out_0, out_1, out_2,
    
         ### Target Vector (7 Features):
    
         # Pitch dummy
         p_1, p_2, p_3, p_4, p_5, p_6, p_7)


### Embedded DNN data:

dnn_emb_model_data <- data %>% 
  arrange(pitcher_id, game_pk) %>% 
  select(### General Information (7 features):
    
         # Game information
         game_year, game_pk, 
         # Situational information
         pitcher_id, pitcher_name, batter_id, batter_name, count,
    
         ### Feature vector (26 features):
    
         # Pitcher conditional probabilities
         p_p1, p_p2, p_p3, p_p4, p_p5, p_p6, p_p7,
         # Batter conditional probabilities
         b_p1, b_p2, b_p3, b_p4, b_p5, b_p6, b_p7,
         # Three previous pitches and locations
         prev_pitch, pp_loc, prev_pitch_2, pp2_loc, prev_pitch_3, pp3_loc,
         # Context variables
         pitch_number, count_code, inning, bat_stand, base_sit, outs_when_up,
    
         ### Target Vector (7 Features):
    
         # Pitch dummy
         p_1, p_2, p_3, p_4, p_5, p_6, p_7)


### One-hot encoded RNN data:

rnn_ohe_model_data <- data %>% 
  arrange(pitcher_id, game_pk) %>% 
  group_by(pitcher_id) %>%
  mutate(sequence_number = cumsum(c(1, diff(pitch_number) <= 0))) %>% 
  ungroup() %>% 
  select(### General Information (8 features):
    
         # Game information
         game_year, game_pk, sequence_number,
         # Situational information
         pitcher_id, pitcher_name, batter_id, batter_name, count,
    
         ### Sequence Data - One-Hot RNN (41 Features):
    
         # Pitch dummy
         p_1, p_2, p_3, p_4, p_5, p_6, p_7,
         # Previous pitch location dummy
         loc_1, loc_2, loc_3, loc_4, loc_5, loc_6, loc_7, 
         loc_8, loc_9, loc_10, loc_11, loc_12, loc_13, loc_14, 
         # Pitch number dummy
         pn_1, pn_2, pn_3, pn_4, pn_5, pn_6, pn_7, pn_8,
         # Count Dummy
         count_00, count_01, count_10, count_11, count_02, count_20, 
         count_12, count_21, count_30, count_31, count_22, count_32,
    
         ### Context Vector (31 Features):
         
         # Inning Dummy
         inn_1, inn_2, inn_3, inn_4, inn_5, inn_6, inn_7, inn_8, inn_9,
         # Batter handedness dummy
         bats_left, bats_right,
         # Base occupation dummy
         b1, b2, b3,
         # Out dummy
         out_0, out_1, out_2,
         # Pitcher conditional probabilities
         p_p1, p_p2, p_p3, p_p4, p_p5, p_p6, p_p7,
         # Batter conditional probabilities
         b_p1, b_p2, b_p3, b_p4, b_p5, b_p6, b_p7)


### Mixed embedding RNN data:

rnn_mix_model_data <- data %>% 
  arrange(pitcher_id, game_pk) %>% 
  group_by(pitcher_id) %>%
  mutate(sequence_number = cumsum(c(1, diff(pitch_number) <= 0))) %>% 
  ungroup() %>% 
  select(### General Information (8 features):
         
         # Game information
         game_year, game_pk, sequence_number,
         # Pitcher and batter information
         pitcher_id, pitcher_name, batter_id, batter_name, count,
         
         ### Sequence Data - Embedded RNN (4 features):
         
         # Pitch information
         pitch, zone, pitch_number, count_code, 
         
         ### Context Vector (51 Features)
         
         # Pitcher conditional probabilities
         p_p1, p_p2, p_p3, p_p4, p_p5, p_p6, p_p7, 
         # Batter conditional probabilities
         b_p1, b_p2, b_p3, b_p4, b_p5, b_p6, b_p7,
         # Pitch number dummy
         pn_1, pn_2, pn_3, pn_4, pn_5, pn_6, pn_7, pn_8,
         # Batter handedness dummy
         bats_left, bats_right,
         # Count Dummy
         count_00, count_01, count_10, count_11, count_02, count_20, 
         count_12, count_21, count_30, count_31, count_22, count_32,
         # Inning Dummy
         inn_1, inn_2, inn_3, inn_4, inn_5, inn_6, inn_7, inn_8, inn_9,
         # Base occupation dummy
         b1, b2, b3,
         # Out dummy
         out_0, out_1, out_2,
         
         ### Target Vector (7 Features):
         
         # Pitch dummy
         p_1, p_2, p_3, p_4, p_5, p_6, p_7)


### Fully embedded RNN data:

rnn_emb_model_data <- data %>% 
  arrange(pitcher_id, game_pk) %>% 
  group_by(pitcher_id) %>%
  mutate(sequence_number = cumsum(c(1, diff(pitch_number) <= 0))) %>% 
  ungroup() %>% 
  select(### General Information (8 features):
    
         # Game information
         game_year, game_pk, sequence_number,
         # Pitcher and batter information
         pitcher_id, pitcher_name, batter_id, batter_name, count,
    
         ### Sequence Data - Embedded RNN (4 features):
    
         # Pitch information 
         pitch, zone, pitch_number, count_code, 
    
         ### Context Vector (18 Features):
    
         # At-bat context
         bat_stand, base_sit, outs_when_up, inning,
         # Pitcher conditional probabilities
         p_p1, p_p2, p_p3, p_p4, p_p5, p_p6, p_p7,
         # Batter conditional probabilities
         b_p1, b_p2, b_p3, b_p4, b_p5, b_p6, b_p7,
    
         ### Target Vector (7 Features):
    
         # Pitch dummy
         p_1, p_2, p_3, p_4, p_5, p_6, p_7)


# Export the Data ---------------------------------------------------------

write.csv(dnn_ohe_model_data, "dnn_ohe_model_data.csv")
write.csv(dnn_emb_model_data, "dnn_emb_model_data.csv")
write.csv(rnn_ohe_model_data, "rnn_ohe_model_data.csv")
write.csv(rnn_mix_model_data, "rnn_mix_model_data.csv")
write.csv(rnn_emb_model_data, "rnn_emb_model_data.csv")

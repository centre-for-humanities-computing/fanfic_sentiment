# %%
library(tidyverse, lme4, lmerTest)

# %%
# load our csv
df <- read_csv("data/merged_sentiment_data.csv")
print(df)

# %%
# print column names
print(colnames(df))
# replace - with _ in column names
colnames(df) <- gsub("-", "_", colnames(df))
print(colnames(df))
# %%
# data pre-processing
df <-  df %>% 
  mutate(words_scaled = scale(words), # sclae word count for better model robustness
         subset = as.factor(subset), # change the levels of the genres for other to be baseline
         subset = fct_relvel(subset, c("other", "angst", "fluff", "hurt/comfort"))) # change this list to be the actual labels in the data

# %%
# we want to know, is there a difference in sentiment between the two groups, fluff and angst? Categories in "subset"
sentiment_col <- "mean_sent_twitter_xlm_roberta_base_sentiment_multilingual" #"mean_sent_syuzhet" #

# fit the model with author as a random effect
model <- lmerTest::lmer(as.formula(paste(sentiment_col, "~ subset + (1 | author)")), data = df)

# suggested model: interaction between genre and fandom, plus contorlling for time and length, and random effect for author
# in other words: do different fandoms have different ways of writing the three genres compared to baseline? 
# model <- lmerTest::lmer(mean_sent_syuzhet ~ subset*fandom_label + published + words_scaled + (1|author), data=df)

# print the summary of the model
summary(model)

# %%
# describe the sentiment column
df %>% 
  summarise(mean = mean(!!sym(sentiment_col), na.rm = TRUE),
            sd = sd(!!sym(sentiment_col), na.rm = TRUE),
            min = min(!!sym(sentiment_col), na.rm = TRUE),
            max = max(!!sym(sentiment_col), na.rm = TRUE),
            n = n())

# %%

col_palette <- c("fluff" = "#6C464F", "hurt/comfort" = "#679436", "other" = "#EF8354",
                 "angst" = "#3B2A4D")

df %>% 
  ggplot(aes_string(x = sentiment_col, color = "subset", fill = "subset")) +
  geom_density(alpha = 0.3) +
  theme_classic() +
  scale_color_manual(values = col_palette) +
  scale_fill_manual(values = col_palette) +
  labs(x = sentiment_col, y = "Density", color = "Subset", fill = "Subset") +
  theme(plot.background = element_rect(color = "black", linewidth = 1))
  # save it
filename <- sprintf("results/figs/R_density_plot_%s.png", sentiment_col)
ggsave(filename, width = 8, height = 6)
# %%
# and a line plot of the means
mean_values <- df %>%
  group_by(subset) %>%
  summarize(mean_sent = mean(mean_sent_syuzhet, na.rm = TRUE))

ggplot(mean_values, aes(x = subset, y = mean_sent, group = 1)) +
  geom_line() +
  geom_point(size = 3) +
  theme_minimal() +
  labs(x = "Subset", y = "Mean Sentiment (syuzhet)", title = "Mean Sentiment by Subset")

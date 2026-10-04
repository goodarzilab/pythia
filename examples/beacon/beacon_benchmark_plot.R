require(ggplot2)
require(cowplot)
require(reshape2)
require(yaml)
require(jsonlite)

# Usage: Rscript beacon_benchmark_plot.R <performance-dir> [outdir]
#
# <performance-dir> is the `common.figure_dir` from pythia_beacon_config.yaml
# (populated by pythia_beacon_structural_tasks.py): it must contain
# cmp_pythia_performance.json, dmp_pythia_performance.json, and
# ssi_pythia_performance.json. [outdir] defaults to <performance-dir>.
# beacon_leaderboard.yaml is read from this script's own directory.

.get_script_dir <- function() {
    cmd_args <- commandArgs(trailingOnly = FALSE)
    file_arg <- "--file="
    match <- grep(file_arg, cmd_args, value = TRUE)
    if (length(match) > 0) {
        return(dirname(normalizePath(sub(file_arg, "", match[1]))))
    }
    getwd()
}

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1) {
    stop("Usage: Rscript beacon_benchmark_plot.R <performance-dir> [outdir]")
}

script_dir <- .get_script_dir()
leaderboard_path <- file.path(script_dir, "beacon_leaderboard.yaml")
pythia_perf_dir <- args[1]
out_dir <- if (length(args) >= 2) args[2] else pythia_perf_dir
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

create_plot <- function(P, path, width, height, make_pdf = TRUE) {
    png(sprintf("%s.png", path), res = 200, width = width * 200, height = height * 200)
    plot(P)
    dev.off()
    if (make_pdf) {
        pdf(sprintf("%s.pdf", path), width = width, height = height)
        plot(P)
        dev.off()
    }
}

# Load leaderboard (all BEACON entries except Pythia)
leaderboard <- read_yaml(leaderboard_path)
in_df <- do.call(rbind, lapply(leaderboard, as.data.frame))
rownames(in_df) <- names(leaderboard)

# Load Pythia's final test-set performance
cmp <- read_json(file.path(pythia_perf_dir, "cmp_pythia_performance.json"))
dmp <- read_json(file.path(pythia_perf_dir, "dmp_pythia_performance.json"))
ssi <- read_json(file.path(pythia_perf_dir, "ssi_pythia_performance.json"))

pythia_row <- data.frame(
    SSP_F1_percent = NA_real_,
    CMP_P.L_percent = cmp$test$test$top_l_precision * 100,
    DMP_R2_percent = dmp$test$test$r2 * 100,
    SSI_R2_percent = ssi$test$r2 * 100,
    row.names = "Pythia"
)

in_df <- rbind(in_df, pythia_row[colnames(in_df)])

# Load and process data
metric_df <- melt(as.matrix(in_df))
# Drop metrics with no value (e.g. Pythia's SSP result is not yet final)
metric_df <- metric_df[!is.na(metric_df$value), ]

# Extract Task and Metric from column names (Var2)
metric_df$Task = unlist(lapply(strsplit(as.character(metric_df$Var2), "_"), function(x) x[[1]]))
metric_df$Metric = unlist(lapply(strsplit(as.character(metric_df$Var2), "_"), function(x) x[[2]]))
# Define main models vs foundation models
main_models = c(
    "Pythia",
    "CNN",
    "ResNet",
    "LSTM"
)
metric_df$SelectModels = ifelse(metric_df$Var1 %in% main_models, as.character(metric_df$Var1), "RNA foundation models")

# Rename specific metrics
metric_df$Metric[metric_df$Metric == "P.L"] = "Precision@L"
metric_df$Facet = paste(metric_df$Task, metric_df$Metric, sep=": ")

# --- Sorting Logic Start ---
# 1. Create a unique identifier for each Model within each Task
metric_df$Var1_Ordered <- paste(metric_df$Var1, metric_df$Task, sep = "___")

# 2. Reorder this new factor based on the 'value' (Low to High is default for reorder)
metric_df$Var1_Ordered <- reorder(metric_df$Var1_Ordered, metric_df$value)

# 3. Helper function to remove the '___Task' suffix from labels during plotting
clean_labels <- function(x) {
  sub("___.*", "", x)
}
# --- Sorting Logic End ---

p = ggplot(metric_df, aes(x=Var1_Ordered, y=value, fill=SelectModels)) +
    geom_bar(stat="identity", position=position_dodge()) +
    # Use scales="free" (or "free_x") to allow different x-axis orders per panel
    facet_wrap(~Facet, scales="free", nrow=1) +
    theme_cowplot() +
    scale_fill_manual(
        values = c("Pythia" = "#1b9e77", "CNN" = "#d95f02", "ResNet" = "#7570b3", "LSTM" = "#e7298a", "RNA foundation models" = "gray")) +
    # Apply the label cleaner to hide the sorting trick
    scale_x_discrete(labels = clean_labels) +
    theme(
        axis.text.x = element_text(angle = 45, hjust = 1),
        axis.title  = element_text(face = "bold"),
        plot.title  = element_text(face = "bold"),
        legend.position = "none"
    ) +
    ggtitle("BEACON Benchmark Results") +
    xlab("Model") # Re-label axis since we are using the composite variable

# Save the plot
create_plot(p, sprintf("%s/beacon_benchmark_results", out_dir), width=16, height=6)

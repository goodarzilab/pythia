options(stringsAsFactors = FALSE)

suppressPackageStartupMessages({
    library(ggplot2)
    library(reshape2)
    library(cowplot)
    library(viridis)
})

# Usage: Rscript sequence_structure_contribution_plot.R <indir> <outdir>
#
# <indir> is the --data-dir passed to compute_deeplift_contributions.py: a
# root directory containing, for each RBP, a
# {indir}/{rbp}/Pythia/{rbp}_deeplift_peak_ratio.tsv.gz file.

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) {
    stop("Usage: Rscript sequence_structure_contribution_plot.R <indir> <outdir>")
}

indir  <- args[1]
outdir <- args[2]

dir.create(outdir, showWarnings = FALSE, recursive = TRUE)

get_inpath <- function(rbp) {
    file.path(indir, rbp, "Pythia", sprintf("%s_deeplift_peak_ratio.tsv.gz", rbp))
}

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

scale_vals <- function(x, y = 1) {
    if (length(y) == 1) y <- x
    s <- sd(y)
    if (is.na(s) || s == 0) return(rep(NA_real_, length(x)))
    (x - mean(y)) / s
}

rbps <- dir(indir)
rbps <- rbps[!grepl("-myc", rbps)]
rbps <- rbps[!grepl("FUS",  rbps)]
rbps <- rbps[!grepl("CLTA", rbps)]
rbps <- rbps[!grepl("tsv",  rbps)]
rbps <- rbps[!grepl("ROC",  rbps)]

list.data <- lapply(rbps, function(rbp) {
    inpath <- get_inpath(rbp)
    if (!file.exists(inpath)) return(NULL)
    message(rbp)
    indf <- read.csv(inpath, sep = "\t", header = TRUE)
    indf$variance.dls <- apply(indf[, c("structure", "seq", "dl")], 1, var)
    num_peaks <- round(nrow(indf) * 0.1)
    if (nrow(indf) > num_peaks) {
        indf <- indf[order(indf$variance.dls, decreasing = TRUE)[seq_len(num_peaks)], ]
    }
    indf$RBP <- rbp
    indf <- indf[!(is.na(indf$log_ratio) | is.infinite(indf$log_ratio)), ]
    for (variable in c("structure", "seq", "dl")) {
        indf[, sprintf("%s.scaled", variable)] <- scale_vals(indf[, variable],
                                                              indf[, "dl"])
    }
    indf$str.minus.dl        <- indf$structure.scaled - indf$dl.scaled
    indf$Median.str.minus.dl <- median(indf$str.minus.dl)
    indf$seq.minus.dl        <- indf$seq.scaled - indf$dl.scaled
    indf$Median.seq.minus.dl <- median(indf$seq.minus.dl)
    indf$log_ratio           <- log2(indf$seq.scaled / indf$structure.scaled)
    indf <- indf[is.finite(indf$log_ratio) & is.finite(indf$str.minus.dl) &
                     is.finite(indf$seq.minus.dl), ]
    if (sum(indf$structure) == 0) {
        cat(sprintf("%s is corrupt\n", rbp))
        return(NULL)
    }
    if (nrow(indf) < 3) {
        cat(sprintf("%s skipped: too few finite observations\n", rbp))
        return(NULL)
    }
    indf$Median.LR           <- median(indf$log_ratio, na.rm = TRUE)
    indf$Max.LR              <- max(indf$log_ratio,    na.rm = TRUE)
    indf$p.value     <- wilcox.test(indf$log_ratio)$p.value
    indf$p.value.str <- wilcox.test(indf$str.minus.dl)$p.value
    indf$p.value.seq <- wilcox.test(indf$seq.minus.dl)$p.value
    indf
})

statdf <- do.call("rbind", list.data)

statdf_path <- file.path(outdir, "Pythia_structure_contribution_plot_data.tsv.gz")
write.table(statdf, gzfile(statdf_path),
            sep = "\t", quote = FALSE, row.names = FALSE)
message(sprintf("Saved plotting dataframe to: %s", statdf_path))

# --- log-ratio boxplot, ordered by median ---
mediandf        <- unique(statdf[, c("RBP", "Median.LR", "p.value")])
mediandf$FDR    <- p.adjust(mediandf$p.value, method = "BH")
mediandf        <- mediandf[order(mediandf$Median.LR, decreasing = TRUE), ]
order_rbps      <- as.character(mediandf$RBP)
statdf$RBP      <- factor(statdf$RBP, levels = order_rbps)

color_rows <- rep("purple", nrow(mediandf))
color_rows[mediandf$Median.LR < -0.05 & mediandf$FDR < 0.1] <- "black"
color_rows[mediandf$Median.LR >  0.05 & mediandf$FDR < 0.1] <- "orange"
mediandf$Color <- color_rows

bold_genes    <- c("SFPQ", "EMG1", "PCBP1", "HNRNPU", "HNRNPC")
fontface_rows <- ifelse(order_rbps %in% bold_genes, "bold", "plain")

color_palette <- "magma"

P <- ggplot(statdf, aes(x = RBP, y = log_ratio, fill = Median.LR)) +
    geom_boxplot(outlier.size = NA, outlier.shape = NA) +
    theme_bw(base_size = 14) +
    scale_fill_viridis(
        name   = expression("Median " * log[2] * " " * frac(
            "DeepLIFT randomized sequence",
            "DeepLIFT randomized structure")),
        option = color_palette) +
    xlab("") +
    ylab(expression(bold(log[2] * " " * frac(
        "DeepLIFT randomized sequence",
        "DeepLIFT randomized structure")))) +
    theme(
        axis.title      = element_text(face = "bold"),
        axis.text.x     = element_text(angle = 45, hjust = 1, color = color_rows,
                                       face = fontface_rows),
        legend.position = "bottom") +
    coord_cartesian(ylim = range(mediandf$Median.LR) * 1.5)

create_plot(P,
    file.path(outdir, sprintf("Pythia_structure_contribution_boxplot_medianOrdered_%s_wilcox", color_palette)),
    width = 22, height = 8)

# --- seq-minus-dl boxplot ---
mediandf <- unique(statdf[, c("RBP", "Median.LR", "p.value",
                               "Median.str.minus.dl", "Median.seq.minus.dl",
                               "p.value.seq", "p.value.str")])
mediandf$FDR     <- p.adjust(mediandf$p.value,     method = "BH")
mediandf$FDR.seq <- p.adjust(mediandf$p.value.seq, method = "BH")
mediandf$FDR.str <- p.adjust(mediandf$p.value.str, method = "BH")
mediandf         <- mediandf[order(mediandf$Median.seq.minus.dl, decreasing = TRUE), ]
order_rbps       <- mediandf$RBP
statdf$RBP       <- factor(statdf$RBP, levels = order_rbps)

color_rows <- rep("purple", nrow(mediandf))
color_rows[mediandf$Median.seq.minus.dl < -0.05 & mediandf$FDR.seq < 5e-2] <- "black"
color_rows[mediandf$Median.seq.minus.dl >  0.05 & mediandf$FDR.seq < 5e-2] <- "orange"

fontface_rows <- ifelse(order_rbps %in% bold_genes, "bold", "plain")

P <- ggplot(statdf, aes(x = RBP, y = seq.minus.dl, fill = Median.seq.minus.dl)) +
    geom_boxplot(outlier.size = NA, outlier.shape = NA) +
    theme_bw(base_size = 14) +
    scale_fill_viridis(name = "", option = color_palette) +
    xlab("") +
    ylab(expression(bold("DeepLIFT randomized sequence - full network"))) +
    theme(
        axis.title      = element_text(face = "bold"),
        axis.text.x     = element_text(angle = 45, hjust = 1, color = color_rows,
                                       face = fontface_rows),
        legend.position = "bottom") +
    coord_cartesian(ylim = range(mediandf$Median.seq.minus.dl) * 1.5)

create_plot(P,
    file.path(outdir, sprintf("Pythia_structure_contribution_boxplot_medianOrdered_%s_wilcox_seq_minus_dl", color_palette)),
    width = 22, height = 8)

message(sprintf("Done. Outputs in: %s", outdir))

# 加载必要的包
library(readxl)
library(ggplot2)
library(dplyr)
library(tidyr)
library(patchwork)
library(stringr)

# 1. 读取数据并重命名列
file_path <- "01.data.xlsx"
data <- read_excel(file_path, skip = 1)
colnames(data) <- c("Species", "Haplotype", "Chr", 
                    "Satellite", "Intact_LTR_RTs", "NUMT", 
                    "NUPT", "rDNA_5S", "Dominant")

# 2. 计算Others比例
data <- data %>%
  mutate(
    Others = 100 - (Satellite + Intact_LTR_RTs + NUMT + NUPT + rDNA_5S),
    Others = pmax(0, Others)
  )

# 3. 标准化物种名称
data <- data %>%
  mutate(
    Species_clean = case_when(
      grepl("japonica", Species, ignore.case = TRUE) ~ "O. sativa ssp. japonica",
      grepl("indica", Species, ignore.case = TRUE) ~ "O. sativa ssp. indica",
      grepl("glaberrima", Species, ignore.case = TRUE) ~ "O. glaberrima",
      grepl("rufipogon", Species, ignore.case = TRUE) ~ "O. rufipogon",
      grepl("nivara", Species, ignore.case = TRUE) ~ "O. nivara",
      grepl("longistaminata", Species, ignore.case = TRUE) ~ "O. longistaminata",
      grepl("glumaepatula|glumaepatule", Species, ignore.case = TRUE) ~ "O. glumaepatule",
      grepl("punctata|punctate", Species, ignore.case = TRUE) ~ "O. punctate",
      grepl("officinalis", Species, ignore.case = TRUE) ~ "O. officinalis",
      grepl("australiensis", Species, ignore.case = TRUE) ~ "O. australiensis",
      grepl("brachyantha", Species, ignore.case = TRUE) ~ "O. brachyantha",
      grepl("meyeriana", Species, ignore.case = TRUE) ~ "O. meyeriana",
      TRUE ~ Species
    ),
    Haplotype = tolower(Haplotype),
    Haplotype = ifelse(grepl("hap", Haplotype), Haplotype, paste0("hap", Haplotype))
  )

# 4. 定义显示顺序（从上到下，不反转）
display_labels <- c(
  # 前2个物种：只显示物种名（hap1）
  "O. sativa ssp. japonica",
  "O. sativa ssp. indica",
  # 后10个物种：每个物种显示hap1和hap2
  "O. glaberrima\nhap1",
  "O. glaberrima\nhap2",
  "O. rufipogon\nhap1",
  "O. rufipogon\nhap2",
  "O. nivara\nhap1",
  "O. nivara\nhap2",
  "O. longistaminata\nhap1",
  "O. longistaminata\nhap2",
  "O. glumaepatule\nhap1",
  "O. glumaepatule\nhap2",
  "O. punctate\nhap1",
  "O. punctate\nhap2",
  "O. officinalis\nhap1",
  "O. officinalis\nhap2",
  "O. australiensis\nhap1",
  "O. australiensis\nhap2",
  "O. brachyantha\nhap1",
  "O. brachyantha\nhap2",
  "O. meyeriana\nhap1",
  "O. meyeriana\nhap2"
)

cat("显示顺序（从上到下）:\n")
for (i in 1:length(display_labels)) {
  cat(sprintf("%2d. %s\n", i, display_labels[i]))
}

# 5. 创建数据映射
data <- data %>%
  mutate(
    Display_Label = case_when(
      Species_clean == "O. sativa ssp. japonica" & Haplotype == "hap1" ~ "O. sativa ssp. japonica",
      Species_clean == "O. sativa ssp. indica" & Haplotype == "hap1" ~ "O. sativa ssp. indica",
      Species_clean == "O. glaberrima" & Haplotype == "hap1" ~ "O. glaberrima\nhap1",
      Species_clean == "O. glaberrima" & Haplotype == "hap2" ~ "O. glaberrima\nhap2",
      Species_clean == "O. rufipogon" & Haplotype == "hap1" ~ "O. rufipogon\nhap1",
      Species_clean == "O. rufipogon" & Haplotype == "hap2" ~ "O. rufipogon\nhap2",
      Species_clean == "O. nivara" & Haplotype == "hap1" ~ "O. nivara\nhap1",
      Species_clean == "O. nivara" & Haplotype == "hap2" ~ "O. nivara\nhap2",
      Species_clean == "O. longistaminata" & Haplotype == "hap1" ~ "O. longistaminata\nhap1",
      Species_clean == "O. longistaminata" & Haplotype == "hap2" ~ "O. longistaminata\nhap2",
      Species_clean == "O. glumaepatule" & Haplotype == "hap1" ~ "O. glumaepatule\nhap1",
      Species_clean == "O. glumaepatule" & Haplotype == "hap2" ~ "O. glumaepatule\nhap2",
      Species_clean == "O. punctate" & Haplotype == "hap1" ~ "O. punctate\nhap1",
      Species_clean == "O. punctate" & Haplotype == "hap2" ~ "O. punctate\nhap2",
      Species_clean == "O. officinalis" & Haplotype == "hap1" ~ "O. officinalis\nhap1",
      Species_clean == "O. officinalis" & Haplotype == "hap2" ~ "O. officinalis\nhap2",
      Species_clean == "O. australiensis" & Haplotype == "hap1" ~ "O. australiensis\nhap1",
      Species_clean == "O. australiensis" & Haplotype == "hap2" ~ "O. australiensis\nhap2",
      Species_clean == "O. brachyantha" & Haplotype == "hap1" ~ "O. brachyantha\nhap1",
      Species_clean == "O. brachyantha" & Haplotype == "hap2" ~ "O. brachyantha\nhap2",
      Species_clean == "O. meyeriana" & Haplotype == "hap1" ~ "O. meyeriana\nhap1",
      Species_clean == "O. meyeriana" & Haplotype == "hap2" ~ "O. meyeriana\nhap2",
      TRUE ~ NA_character_
    )
  )

# 6. 过滤数据
plot_data <- data %>%
  filter(!is.na(Display_Label) & Display_Label %in% display_labels)

# 7. 转换为长格式
data_long <- plot_data %>%
  select(Display_Label, Chr, Satellite, Intact_LTR_RTs, NUMT, NUPT, rDNA_5S, Others) %>%
  pivot_longer(
    cols = c(Satellite, Intact_LTR_RTs, NUMT, NUPT, rDNA_5S, Others),
    names_to = "Sequence_Type",
    values_to = "Percentage"
  ) %>%
  mutate(
    Sequence_Type = factor(Sequence_Type,
                          levels = c("Satellite", "Intact_LTR_RTs", "NUMT", 
                                   "NUPT", "rDNA_5S", "Others"),
                          labels = c("Satellite", "Intact LTR-RTs", "NUMT", 
                                   "NUPT", "5S rDNA", "Others"))
  )

# 8. 设置颜色
my_colors <- c(
  "Satellite" = "#D1161D",
  "Intact LTR-RTs" = "#1272B0",
  "NUMT" = "#F7B472",
  "NUPT" = "#8A5349",
  "5S rDNA" = "#2D8B36",
  "Others" = "#C7C7C7"
)

# 9. 确保顺序（不反转！）
data_long$Display_Label <- factor(data_long$Display_Label, levels = display_labels)  # 不反转！
data_long$Chr <- factor(data_long$Chr, levels = paste0("Chr", sprintf("%02d", 1:12)))

# 10. 验证顺序
cat("\n数据中的显示标签顺序:\n")
print(levels(data_long$Display_Label))

# 11. 创建绘图函数
create_pie <- function(df) {
  if (nrow(df) == 0) {
    return(ggplot() + theme_void() + theme(plot.margin = margin(2, 2, 2, 2)))
  }
  
  ggplot(df, aes(x = "", y = Percentage, fill = Sequence_Type)) +
    geom_bar(stat = "identity", width = 1, color = "white", linewidth = 0.1) +
    coord_polar("y", start = 0) +
    scale_fill_manual(values = my_colors) +
    theme_void() +
    theme(
      plot.margin = margin(2, 2, 2, 2),
      legend.position = "none"
    )
}

# 12. 创建所有组合的饼图
all_chr <- levels(data_long$Chr)
all_labels <- levels(data_long$Display_Label)

plot_list <- list()
for (i in seq_along(all_labels)) {
  for (j in seq_along(all_chr)) {
    cell_data <- data_long %>%
      filter(Display_Label == all_labels[i], Chr == all_chr[j])
    
    p <- create_pie(cell_data)
    plot_list[[paste(i, j, sep = "_")]] <- p
  }
}

# 13. 创建行标签
row_labels <- lapply(all_labels, function(label) {
  ggplot() +
    annotate("text", x = 0, y = 0.5, label = label, 
             size = 3.0, hjust = 0, vjust = 0.5, 
             lineheight = 0.8) +
    theme_void() +
    theme(plot.margin = margin(0, 5, 0, 0))
})

# 14. 创建列标签
col_labels <- lapply(all_chr, function(chr) {
  ggplot() +
    annotate("text", x = 0.5, y = 0.5, label = chr, 
             size = 3.5, hjust = 0.5, vjust = 0.5,
             fontface = "bold") +
    theme_void() +
    theme(plot.margin = margin(0, 0, 5, 0))
})

# 15. 布局组合
empty_corner <- ggplot() + theme_void()

# 列标题
col_header <- wrap_plots(
  c(list(empty_corner), col_labels),
  nrow = 1,
  widths = c(2, rep(1, length(all_chr)))
)

# 数据行
data_rows <- list()
for (i in seq_along(all_labels)) {
  start_idx <- (i - 1) * length(all_chr) + 1
  end_idx <- i * length(all_chr)
  
  row_plots <- c(
    list(row_labels[[i]]),
    plot_list[start_idx:end_idx]
  )
  
  data_row <- wrap_plots(row_plots, nrow = 1, widths = c(2.5, rep(1, length(all_chr))))
  data_rows[[i]] <- data_row
}

main_grid <- wrap_plots(data_rows, ncol = 1)
full_grid <- col_header / main_grid +
  plot_layout(heights = c(0.5, length(all_labels)))

# 16. 创建图例
legend_plot <- ggplot(data_long, aes(x = "", y = Percentage, fill = Sequence_Type)) +
  geom_bar(stat = "identity", width = 1, color = "white", linewidth = 0.1) +
  scale_fill_manual(values = my_colors, name = "Sequence Type") +
  theme_minimal() +
  theme(
    legend.key.size = unit(0.6, "cm"),
    legend.text = element_text(size = 10),
    legend.title = element_text(size = 11, face = "bold")
  )

legend <- cowplot::get_legend(legend_plot)

# 17. 最终组合
final_plot <- (full_grid | legend) + 
  plot_layout(widths = c(10, 1.5))

# 18. 保存PDF
pdf_width <- length(all_chr) * 1.0 + 6
pdf_height <- length(all_labels) * 0.7 + 4

pdf_file <- "Centromere_Composition_Correct_Order.pdf"
ggsave(pdf_file, final_plot, 
       width = pdf_width, height = pdf_height, 
       device = "pdf", dpi = 300, bg = "white")

cat(sprintf("\n✓ PDF已保存: %s\n", pdf_file))
cat(sprintf("尺寸: %.1f × %.1f 英寸\n", pdf_width, pdf_height))
cat("布局: 22行 × 12列\n")
cat("顺序: 从上到下为:\n")
cat("  1. O. sativa ssp. japonica\n")
cat("  2. O. sativa ssp. indica\n")
cat("  3. O. glaberrima hap1\n")
cat("  4. O. glaberrima hap2\n")
cat("  ... 以此类推\n")
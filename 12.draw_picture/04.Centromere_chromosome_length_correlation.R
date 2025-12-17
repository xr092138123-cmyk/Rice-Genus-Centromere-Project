# Load necessary libraries
library(ggplot2)
library(dplyr)

# Read data
centromere_data <- read.csv("fai_summary.csv", header = TRUE)

# 检查数据结构和前几行
cat("数据维度:", dim(centromere_data), "\n")
cat("列名:", names(centromere_data), "\n")
print(head(centromere_data))

# 检查NA值
cat("\n各列的NA值数量:\n")
print(colSums(is.na(centromere_data)))

# 数据转换
centromere_data <- centromere_data %>%
  mutate(
    len_mb = Chromosome.length / 1e6,  # 转换为MB单位
    cen_len_kb = Centromere.length / 1e3,  # 转换为KB单位
    Species_group = gsub("_hap[1234]$", "", Species)  # 移除_hap1/_hap2/_hap3/_hap4后缀
  )

# 检查转换后的数据
cat("\n转换后的数据前几行:\n")
print(head(centromere_data))

# 检查转换后是否有NA
cat("\n转换后各列的NA值数量:\n")
print(colSums(is.na(centromere_data)))

# 定义物种组和颜色映射（使用更新后的物种名）
species_groups <- c("AA_Osat_jap", "AA_Osat_ind", "AA_Ogla", "AA_Oruf", 
                    "AA_Oniv", "AA_Olon", "AA_Oglu", "BB_Opun", 
                    "CC_Ooff", "EE_Oaus", "FF_Obra", "GG_Omey", "L.hexandra")

color_mapping <- c(
  'AA_Osat_jap' = "#59AC6E",
  'AA_Osat_ind' = "#CBE54E",
  'AA_Ogla' = "#76D273",
  'AA_Oruf' = "#215A20",
  'AA_Oniv' = "#3BA738",
  'AA_Olon' = "#51C54E",
  'AA_Oglu' = "#3D8347",
  'BB_Opun' = "#F2AE2C",
  'CC_Ooff' = "#684E94",
  'EE_Oaus' = "#4E84C3",
  'FF_Obra' = "#D55F6F",
  'GG_Omey' = "#9D5427",
  'L.hexandra' = "#595959"
)

# 将Species_group转换为因子并设置水平顺序
centromere_data$Species_group <- factor(centromere_data$Species_group, levels = species_groups)

# 检查Species_group是否有不在定义中的值
cat("\nSpecies_group的值分布:\n")
print(table(centromere_data$Species_group, useNA = "always"))

# 移除所有NA值（包括len_mb, cen_len_kb和Species_group）
centromere_clean <- centromere_data %>% 
  filter(!is.na(len_mb) & !is.na(cen_len_kb) & !is.na(Species_group))

cat("\n清理后的数据维度:", dim(centromere_clean), "\n")

# 检查清理后数据的物种分布
cat("\n清理后Species_group的值分布:\n")
print(table(centromere_clean$Species_group))

# 计算线性回归的统计量（使用清理后的数据）
lm_model <- lm(cen_len_kb ~ len_mb, data = centromere_clean)
r_squared <- summary(lm_model)$r.squared
p_value <- summary(lm_model)$coefficients[2, 4]

# 格式化统计标签
stat_label <- paste0("R² = ", round(r_squared, 3), 
                    ", p = ", ifelse(p_value < 0.001, "< 0.001", round(p_value, 3)))

# Create the plot (使用清理后的数据)
correlation_plot <- ggplot(centromere_clean, aes(x = len_mb, y = cen_len_kb)) +
  geom_point(aes(color = Species_group), size = 3, alpha = 0.7) +
  geom_smooth(method = 'lm', color = "black") +
  annotate("text", x = Inf, y = Inf, 
           label = stat_label, 
           hjust = 1.1, vjust = 1.1, 
           size = 4, color = "black") +
  scale_color_manual(
    name = "Species Group",
    values = color_mapping,
    labels = c(
      "AA_Osat_jap" = "O. sativa ssp. japonica",
      "AA_Osat_ind" = "O. sativa ssp. indica", 
      "AA_Ogla" = "O. glaberrima",
      "AA_Oruf" = "O. rufipogon",
      "AA_Oniv" = "O. nivara",
      "AA_Olon" = "O. longistaminata",
      "AA_Oglu" = "O. glumaepatula",
      "BB_Opun" = "O. punctata",
      "CC_Ooff" = "O. officinalis",
      "EE_Oaus" = "O. australiensis",
      "FF_Obra" = "O. brachyantha",
      "GG_Omey" = "O. meyeriana",
      "L.hexandra" = "L. hexandra"
    )
  ) +
  labs(
    title = "Correlation between Chromosome Length and Centromere Size",
    x = "Chromosome length (Mb)",
    y = "Centromere length (kb)"
  ) +
  theme_bw() +
  theme(
    legend.position = "bottom",
    legend.text = element_text(size = 8),
    legend.title = element_text(size = 10),
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5),
    legend.key.size = unit(0.5, "cm")
  ) +
  guides(color = guide_legend(nrow = 4, byrow = TRUE))  # 图例分4行显示

# 显示图形
print(correlation_plot)

# Save as PDF with high quality (宽度减半，高度适当增加以容纳图例)
ggsave(
  filename = "C:\\Users\\李祥瑞\\Desktop\\Chromosome_vs_Centromere_length.pdf",
  plot = correlation_plot,
  width = 6,
  height = 8,  # 稍微减少高度，因为宽度减半后图形比例更好
  device = "pdf",
  dpi = 300
)

# Statistical summary (使用清理后的数据)
summary_stats <- centromere_clean %>%
  group_by(Species_group) %>%
  summarise(
    mean_Centromere_length_kb = mean(cen_len_kb, na.rm = TRUE),
    sd_Centromere_length_kb = sd(cen_len_kb, na.rm = TRUE),
    mean_Chromosome_length_Mb = mean(len_mb, na.rm = TRUE),
    sd_Chromosome_length_Mb = sd(len_mb, na.rm = TRUE),
    n = n()  # 添加样本量统计
  )

# Print summary statistics
cat("\n描述性统计:\n")
print(summary_stats)

# 打印回归分析结果
cat("\n回归分析结果:\n")
cat("R-squared:", round(r_squared, 4), "\n")
cat("P-value:", format.pval(p_value, digits = 3), "\n")
package processors

import (
	"fmt"
	"log"
	"regexp"
	"strings"
	"time"
	"unicode"

	"videoSummarize/core"
)

// TextAligner 文本对齐处理器
type TextAligner struct {
	// 配置参数
	MaxEditDistance int     // 最大编辑距离
	SimilarityThreshold float64 // 相似度阈值
}

// NewTextAligner 创建新的文本对齐处理器
func NewTextAligner() *TextAligner {
	return &TextAligner{
		MaxEditDistance:     10,
		SimilarityThreshold: 0.6,
	}
}

// AlignmentResult 对齐结果
type AlignmentResult struct {
	AlignedSegments []core.Segment `json:"aligned_segments"`
	QualityScore    float64        `json:"quality_score"`
	ProcessingTime  int64          `json:"processing_time_ms"`
	Errors          []string       `json:"errors,omitempty"`
}

// PreprocessText 预处理阶段：空格规范化处理
func (ta *TextAligner) PreprocessText(text string) string {
	log.Printf("Starting text preprocessing")
	
	// 1. 去除多余空格
	text = regexp.MustCompile(`\s+`).ReplaceAllString(text, " ")
	
	// 2. 去除首尾空格
	text = strings.TrimSpace(text)
	
	// 3. 规范化标点符号周围的空格
	text = regexp.MustCompile(`\s*([，。！？；：])\s*`).ReplaceAllString(text, "$1")
	
	// 4. 去除连续的标点符号
	repeatedPunctRe := regexp.MustCompile(`([，。！？；：])+`)
	text = repeatedPunctRe.ReplaceAllString(text, "$1")
	
	log.Printf("Text preprocessing completed")
	return text
}

// SplitTextUnits 文本分割：分割为句子和单词级别的单元
func (ta *TextAligner) SplitTextUnits(text string) ([]string, []string) {
	log.Printf("Starting text unit splitting")
	
	// 分割为句子（基于标点符号）
	sentenceRegex := regexp.MustCompile(`[^，。！？；：]+[，。！？；：]?`)
	sentences := sentenceRegex.FindAllString(text, -1)
	
	// 清理句子
	var cleanSentences []string
	for _, sentence := range sentences {
		sentence = strings.TrimSpace(sentence)
		if sentence != "" {
			cleanSentences = append(cleanSentences, sentence)
		}
	}
	
	// 分割为单词
	var words []string
	for _, sentence := range cleanSentences {
		// 按空格和标点符号分割
		wordRegex := regexp.MustCompile(`[\p{L}\p{N}]+`)
		sentenceWords := wordRegex.FindAllString(sentence, -1)
		words = append(words, sentenceWords...)
	}
	
	log.Printf("Text splitting completed: %d sentences, %d words", len(cleanSentences), len(words))
	return cleanSentences, words
}

// EditDistance 计算编辑距离（Levenshtein距离）
func (ta *TextAligner) EditDistance(s1, s2 string) int {
	runes1 := []rune(s1)
	runes2 := []rune(s2)
	len1, len2 := len(runes1), len(runes2)
	
	// 创建DP表
	dp := make([][]int, len1+1)
	for i := range dp {
		dp[i] = make([]int, len2+1)
	}
	
	// 初始化边界条件
	for i := 0; i <= len1; i++ {
		dp[i][0] = i
	}
	for j := 0; j <= len2; j++ {
		dp[0][j] = j
	}
	
	// 填充DP表
	for i := 1; i <= len1; i++ {
		for j := 1; j <= len2; j++ {
			if runes1[i-1] == runes2[j-1] {
				dp[i][j] = dp[i-1][j-1]
			} else {
				dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
			}
		}
	}
	
	return dp[len1][len2]
}

// min 返回三个数中的最小值
func min(a, b, c int) int {
	if a <= b && a <= c {
		return a
	}
	if b <= c {
		return b
	}
	return c
}

// CalculateSimilarity 计算文本相似度
func (ta *TextAligner) CalculateSimilarity(s1, s2 string) float64 {
	if s1 == s2 {
		return 1.0
	}
	
	maxLen := len(s1)
	if len(s2) > maxLen {
		maxLen = len(s2)
	}
	
	if maxLen == 0 {
		return 1.0
	}
	
	editDist := ta.EditDistance(s1, s2)
	return 1.0 - float64(editDist)/float64(maxLen)
}

// AlignTexts 对齐处理：使用编辑距离算法进行文本对齐
func (ta *TextAligner) AlignTexts(originalSegments []core.Segment, correctedText string) ([]core.Segment, error) {
	log.Printf("Starting text alignment for %d segments", len(originalSegments))
	
	// 预处理修正文本
	processedCorrectedText := ta.PreprocessText(correctedText)
	
	// 分割修正文本
	correctedSentences, _ := ta.SplitTextUnits(processedCorrectedText)
	
	// 如果修正文本句子数量与原始片段数量不匹配，尝试智能分割
	if len(correctedSentences) != len(originalSegments) {
		log.Printf("Sentence count mismatch: %d corrected vs %d original, attempting smart alignment", 
			len(correctedSentences), len(originalSegments))
		correctedSentences = ta.SmartSplit(processedCorrectedText, len(originalSegments))
	}
	
	// 创建对齐结果
	alignedSegments := make([]core.Segment, len(originalSegments))
	
	// 逐个对齐片段
	for i, originalSegment := range originalSegments {
		if i < len(correctedSentences) {
			// 计算相似度
			originalText := ta.PreprocessText(originalSegment.Text)
			similarity := ta.CalculateSimilarity(originalText, correctedSentences[i])
			
			// 如果相似度过低，保留原文本
			if similarity < ta.SimilarityThreshold {
				log.Printf("Low similarity (%.2f) for segment %d, keeping original text", similarity, i)
				alignedSegments[i] = originalSegment
			} else {
				alignedSegments[i] = core.Segment{
					Start: originalSegment.Start,
					End:   originalSegment.End,
					Text:  correctedSentences[i],
				}
			}
		} else {
			// 如果修正文本不够，保留原文本
			alignedSegments[i] = originalSegment
		}
	}
	
	log.Printf("Text alignment completed")
	return alignedSegments, nil
}

// SmartSplit 智能分割文本以匹配目标片段数量
func (ta *TextAligner) SmartSplit(text string, targetCount int) []string {
	log.Printf("Performing smart split to %d segments", targetCount)
	
	// 首先按标点符号分割
	sentences, _ := ta.SplitTextUnits(text)
	
	if len(sentences) == targetCount {
		return sentences
	}
	
	if len(sentences) > targetCount {
		// 如果句子太多，合并相邻句子
		return ta.MergeSentences(sentences, targetCount)
	} else {
		// 如果句子太少，分割长句子
		return ta.SplitSentences(sentences, targetCount)
	}
}

// MergeSentences 合并句子以达到目标数量
func (ta *TextAligner) MergeSentences(sentences []string, targetCount int) []string {
	if len(sentences) <= targetCount {
		return sentences
	}
	
	result := make([]string, targetCount)
	sentencesPerGroup := len(sentences) / targetCount
	extrasentences := len(sentences) % targetCount
	
	sentenceIndex := 0
	for i := 0; i < targetCount; i++ {
		groupSize := sentencesPerGroup
		if i < extrasentences {
			groupSize++
		}
		
		var groupSentences []string
		for j := 0; j < groupSize && sentenceIndex < len(sentences); j++ {
			groupSentences = append(groupSentences, sentences[sentenceIndex])
			sentenceIndex++
		}
		
		result[i] = strings.Join(groupSentences, "")
	}
	
	return result
}

// SplitSentences 分割句子以达到目标数量
func (ta *TextAligner) SplitSentences(sentences []string, targetCount int) []string {
	if len(sentences) >= targetCount {
		return sentences[:targetCount]
	}
	
	result := make([]string, 0, targetCount)
	neededSplits := targetCount - len(sentences)
	
	for _, sentence := range sentences {
		if neededSplits > 0 && len(sentence) > 20 { // 只分割较长的句子
			// 简单按长度分割
			midPoint := len(sentence) / 2
			// 寻找最近的空格或标点符号
			for i := midPoint; i < len(sentence) && i > 0; i++ {
				if unicode.IsSpace(rune(sentence[i])) || unicode.IsPunct(rune(sentence[i])) {
					midPoint = i
					break
				}
			}
			
			result = append(result, strings.TrimSpace(sentence[:midPoint]))
			result = append(result, strings.TrimSpace(sentence[midPoint:]))
			neededSplits--
		} else {
			result = append(result, sentence)
		}
	}
	
	// 如果还不够，用空字符串填充
	for len(result) < targetCount {
		result = append(result, "")
	}
	
	return result[:targetCount]
}

// ValidateAlignment 后处理：对对齐结果进行质量验证
func (ta *TextAligner) ValidateAlignment(originalSegments, alignedSegments []core.Segment) (float64, []string) {
	log.Printf("Starting alignment validation")
	
	var errors []string
	totalSimilarity := 0.0
	validSegments := 0
	
	// 检查片段数量
	if len(originalSegments) != len(alignedSegments) {
		errors = append(errors, fmt.Sprintf("Segment count mismatch: %d vs %d", 
			len(originalSegments), len(alignedSegments)))
	}
	
	// 检查每个片段
	for i := 0; i < len(originalSegments) && i < len(alignedSegments); i++ {
		original := originalSegments[i]
		aligned := alignedSegments[i]
		
		// 检查时间戳
		if original.Start != aligned.Start || original.End != aligned.End {
			errors = append(errors, fmt.Sprintf("Timestamp mismatch in segment %d", i))
		}
		
		// 计算文本相似度
		if aligned.Text != "" {
			similarity := ta.CalculateSimilarity(original.Text, aligned.Text)
			totalSimilarity += similarity
			validSegments++
			
			// 检查是否有明显的质量问题
			if similarity < 0.3 {
				errors = append(errors, fmt.Sprintf("Very low similarity (%.2f) in segment %d", similarity, i))
			}
		}
	}
	
	// 计算平均质量分数
	qualityScore := 0.0
	if validSegments > 0 {
		qualityScore = totalSimilarity / float64(validSegments)
	}
	
	log.Printf("Alignment validation completed: quality score %.2f, %d errors", qualityScore, len(errors))
	return qualityScore, errors
}

// ProcessAlignment 完整的文本对齐处理流程
func (ta *TextAligner) ProcessAlignment(originalSegments []core.Segment, correctedText string) (*AlignmentResult, error) {
	startTime := getCurrentTimeMillis()
	log.Printf("Starting complete alignment process")
	
	// 1. 预处理
	processedText := ta.PreprocessText(correctedText)
	
	// 2. 对齐处理
	alignedSegments, err := ta.AlignTexts(originalSegments, processedText)
	if err != nil {
		return nil, fmt.Errorf("alignment failed: %v", err)
	}
	
	// 3. 后处理验证
	qualityScore, errors := ta.ValidateAlignment(originalSegments, alignedSegments)
	
	processingTime := getCurrentTimeMillis() - startTime
	
	result := &AlignmentResult{
		AlignedSegments: alignedSegments,
		QualityScore:    qualityScore,
		ProcessingTime:  processingTime,
		Errors:          errors,
	}
	
	log.Printf("Alignment process completed in %dms with quality score %.2f", processingTime, qualityScore)
	return result, nil
}

// getCurrentTimeMillis 获取当前时间的毫秒数
func getCurrentTimeMillis() int64 {
	return time.Now().UnixNano() / int64(time.Millisecond)
}
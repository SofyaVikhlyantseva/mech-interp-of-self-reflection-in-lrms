import json
import re
from ollama import chat
from tqdm import tqdm
import time
from datetime import datetime

# ========== КОНФИГУРАЦИЯ ==========
JUDGE_MODEL = 'qwen3:8b'  # Модель-судья (llm-as-a-judge)
RESULTS_FILE = 'merged_results.json'
OUTPUT_FILE = 'classified_corrections.json'
STATS_FILE = 'classification_stats.json'

# Чекпоинт для возобновления (None = начинать с нуля)
RESUME_FROM_CHECKPOINT = 'checkpoint_classified_90.json'

REFLECTION_PHRASES = [
    'wait', 'actually', 'let me reconsider', 'correction',
    'sorry', 'i meant', 'my mistake', 'on second thought',
    'i should re-examine', 'that seems wrong', 'i need to correct',
    'hang on', 'hold on', 'that\'s not right', 'i made an error'
]

def log_message(msg):
    """Логирует сообщение с временной меткой"""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    print(full_msg)

def tokenize_simple(text):
    """
    Простая токенизация (разбиение по пробелам и пунктуации).
    """
    tokens = re.findall(r'\w+|[^\w\s]', text.lower())
    return tokens

def find_marker_positions(tokens, marker):
    """
    Находит все позиции маркера в списке токенов.

    Args:
        tokens: список токенов
        marker: маркер для поиска (например, "wait")

    Returns:
        Список индексов позиций маркера
    """
    marker_lower = marker.lower()
    positions = []

    for i, token in enumerate(tokens):
        if token == marker_lower:
            positions.append(i)

    return positions

def extract_context(text, marker, window=50):
    """
    Извлекает контекст вокруг маркера:
    окно токенов до + маркер + окно токенов после

    Args:
        text: полный текст thinking
        marker: маркер самоисправления (например, "wait")
        window: количество токенов до/после (по умолчанию 50)

    Returns:
        Список контекстов (может быть несколько вхождений маркера)
    """
    if not text:
        return []

    tokens = tokenize_simple(text)
    marker_positions = find_marker_positions(tokens, marker)

    contexts = []

    for position in marker_positions:
        # Границы контекста
        start = max(0, position - window)
        end = min(len(tokens), position + window + 1)

        # Извлекаем части
        before_tokens = tokens[start:position]
        after_tokens = tokens[position+1:end]
        context_tokens = tokens[start:end]

        contexts.append({
            'position': position,
            'before': ' '.join(before_tokens),
            'marker': marker.lower(),
            'after': ' '.join(after_tokens),
            'full_context': ' '.join(context_tokens),
            'context_length': len(context_tokens)
        })

    return contexts

def classify_correction(context_before, marker, context_after, question=None, max_retries=3):
    """
    Использует LLM-as-a-judge для классификации маркера.
    """

    # Формируем промпт для судьи
    judge_prompt = f"""You are analyzing whether a self-correction marker represents a GENUINE CORRECTION or STYLISTIC FILLER.

**Original question:**
{question[:200] if question else 'Not provided'}...

**Context BEFORE "{marker}":**
{context_before}

**→ MARKER: "{marker}" ←**

**Context AFTER:**
{context_after}

**CLASSIFICATION CRITERIA:**

1. **genuine_correction**: Model realizes error and CHANGES reasoning/conclusion
   - Look for: contradicting previous statement, different answer, "I was wrong"
   - Example: "Wait, that's incorrect. The answer should be X" (previously said Y)

2. **stylistic_filler**: Model uses "{marker}" without actual change
   - Look for: continues same reasoning, just elaborating, rhetorical pause
   - Example: "Wait, let me think about this more carefully..." (then same approach)

3. **unclear**: Cannot determine confidently
   - Use when context is ambiguous or too short

**CONFIDENCE CRITERIA:**

- **high**: Clear evidence for classification
  * For genuine: Explicit contradiction (e.g., "that's wrong", opposite conclusion)
  * For stylistic: Clear continuation of same reasoning without any change
  * Context is sufficient (both before/after are meaningful)

- **medium**: Reasonable inference but not explicit
  * For genuine: Implicit change (different approach but no explicit "I was wrong")
  * For stylistic: Seems like elaboration but could be subtle correction
  * Context is partially limited

- **low**: Weak evidence or insufficient context
  * Very short context (< 10 tokens before or after)
  * Ambiguous whether change occurred
  * Marker at very beginning/end of thinking

**EXAMPLES:**

HIGH confidence genuine:
Before: "The answer is 5"
Marker: "wait"
After: "that's wrong. The answer is actually 10"

HIGH confidence stylistic:
Before: "Let me calculate this step by step"
Marker: "wait"
After: "I'll approach it systematically. First, ..."

MEDIUM confidence genuine:
Before: "Using method A"
Marker: "actually"
After: "let's try method B instead" (no explicit "A was wrong")

LOW confidence:
Before: "So" (too short!)
Marker: "wait"
After: "let me think" (unclear what follows)

**OUTPUT FORMAT (JSON only):**
{{
    "classification": "genuine_correction" OR "stylistic_filler" OR "unclear",
    "confidence": "high" OR "medium" OR "low",
    "reasoning": "Brief explanation (max 100 chars)",
    "evidence": "Key phrase that determined classification"
}}

Analyze carefully and choose confidence based on STRENGTH OF EVIDENCE."""

    for attempt in range(max_retries):
        try:
            response = chat(
                model=JUDGE_MODEL,
                messages=[{'role': 'user', 'content': judge_prompt}],
                format='json',  # Требуем JSON ответ
                stream=False,
                options={'temperature': 0.1}  # Низкая температура для консистентности
            )

            # Извлекаем и парсим JSON
            content = response.message.content.strip()

            # Убираем возможные markdown обертки
            if content.startswith('```'):
                content = re.sub(r'^```(?:json)?\n', '', content)
                content = re.sub(r'\n```$', '', content)

            result = json.loads(content)

            # Валидация обязательных полей
            required_fields = ['classification', 'confidence', 'reasoning']
            if all(field in result for field in required_fields):
                # Нормализуем classification
                cls = result['classification'].lower()
                if cls not in ['genuine_correction', 'stylistic_filler', 'unclear']:
                    raise ValueError(f"Invalid classification: {cls}")

                result['classification'] = cls
                return result
            else:
                log_message(f"Missing fields in response, retrying...")

        except json.JSONDecodeError as e:
            log_message(f"JSON parse error: {str(e)[:50]}, retrying...")
        except Exception as e:
            log_message(f"Error: {str(e)[:50]}, retrying...")

        # Пауза перед повтором
        if attempt < max_retries - 1:
            time.sleep(2)

    # Если все попытки неудачны
    return {
        'classification': 'unclear',
        'confidence': 'low',
        'reasoning': 'Failed to get valid response from judge'
    }

def load_checkpoint(checkpoint_file):
    """
    Загружает данные из чекпоинта.

    Returns:
        (classified_results, processed_ids, processed_markers_count)
        - classified_results: список уже классифицированных вопросов
        - processed_ids: множество ID уже обработанных вопросов
        - processed_markers_count: количество уже обработанных маркеров
    """
    try:
        with open(checkpoint_file, 'r', encoding='utf-8') as f:
            checkpoint_data = json.load(f)

        classified_results = checkpoint_data.get('questions', [])
        processed_markers_count = checkpoint_data.get('processed_markers', 0)

        # Собираем ID уже обработанных вопросов
        processed_ids = {q['id'] for q in classified_results}

        log_message(f"Чекпоинт загружен: {checkpoint_file}")
        log_message(f"  Уже обработано вопросов: {len(classified_results)}")
        log_message(f"  Уже обработано маркеров: {processed_markers_count}")
        log_message(f"  ID обработанных вопросов: {sorted(processed_ids)}")

        return classified_results, processed_ids, processed_markers_count

    except FileNotFoundError:
        log_message(f"Чекпоинт {checkpoint_file} не найден, начинаем с нуля")
        return [], set(), 0
    except json.JSONDecodeError as e:
        log_message(f"ОШИБКА чтения чекпоинта: {e}, начинаем с нуля")
        return [], set(), 0

def generate_classification_stats(results):
    """
    Генерирует статистику по классификации.
    """
    log_message("\nГенерирую статистику...")

    total_markers = 0
    genuine_count = 0
    stylistic_count = 0
    unclear_count = 0

    # Статистика по маркерам
    marker_stats = {}

    # Статистика по категориям вопросов
    category_stats = {}

    confidence_distribution = {'high': 0, 'medium': 0, 'low': 0}

    for question in results:
        category = question.get('category', 'unknown')

        if category not in category_stats:
            category_stats[category] = {
                'total': 0,
                'genuine': 0,
                'stylistic': 0,
                'unclear': 0
            }

        for marker_data in question['markers']:
            marker = marker_data['marker']

            if marker not in marker_stats:
                marker_stats[marker] = {
                    'total': 0,
                    'genuine': 0,
                    'stylistic': 0,
                    'unclear': 0,
                    'confidence_high': 0,
                    'confidence_medium': 0,
                    'confidence_low': 0
                }

            for classification in marker_data['classifications']:
                total_markers += 1
                marker_stats[marker]['total'] += 1
                category_stats[category]['total'] += 1

                cls = classification['classification']
                conf = classification['confidence']

                # Подсчёт по типу классификации
                if cls == 'genuine_correction':
                    genuine_count += 1
                    marker_stats[marker]['genuine'] += 1
                    category_stats[category]['genuine'] += 1
                elif cls == 'stylistic_filler':
                    stylistic_count += 1
                    marker_stats[marker]['stylistic'] += 1
                    category_stats[category]['stylistic'] += 1
                else:  # unclear
                    unclear_count += 1
                    marker_stats[marker]['unclear'] += 1
                    category_stats[category]['unclear'] += 1

                # Подсчёт по confidence
                confidence_distribution[conf] += 1
                marker_stats[marker][f'confidence_{conf}'] += 1

    # ========== ВЫВОД СТАТИСТИКИ ==========
    log_message("\n" + "=" * 70)
    log_message("ИТОГОВАЯ СТАТИСТИКА КЛАССИФИКАЦИИ")
    log_message("=" * 70)

    log_message(f"\nОБЩАЯ СТАТИСТИКА:")
    log_message(f"Всего маркеров: {total_markers}")
    log_message(f"Genuine corrections: {genuine_count} ({100*genuine_count/total_markers:.1f}%)")
    log_message(f"Stylistic fillers: {stylistic_count} ({100*stylistic_count/total_markers:.1f}%)")
    log_message(f"Unclear: {unclear_count} ({100*unclear_count/total_markers:.1f}%)")

    log_message(f"\nCONFIDENCE DISTRIBUTION:")
    for conf, count in confidence_distribution.items():
        log_message(f"  {conf.capitalize()}: {count} ({100*count/total_markers:.1f}%)")

    log_message(f"\nСТАТИСТИКА ПО МАРКЕРАМ:")
    sorted_markers = sorted(marker_stats.items(), key=lambda x: x[1]['total'], reverse=True)

    for marker, stats in sorted_markers:
        total = stats['total']
        genuine = stats['genuine']
        stylistic = stats['stylistic']
        unclear = stats['unclear']

        genuine_pct = 100 * genuine / total if total > 0 else 0
        stylistic_pct = 100 * stylistic / total if total > 0 else 0

        log_message(f"\n  '{marker}' ({total} вхождений):")
        log_message(f"Genuine: {genuine} ({genuine_pct:.1f}%)")
        log_message(f"Stylistic: {stylistic} ({stylistic_pct:.1f}%)")
        log_message(f"Unclear: {unclear}")

    log_message(f"\nТАТИСТИКА ПО КАТЕГОРИЯМ ВОПРОСОВ:")
    for category, stats in sorted(category_stats.items()):
        total = stats['total']
        if total == 0:
            continue

        genuine = stats['genuine']
        stylistic = stats['stylistic']

        log_message(f"\n  {category.upper()} ({total} маркеров):")
        log_message(f"Genuine: {genuine} ({100*genuine/total:.1f}%)")
        log_message(f"Stylistic: {stylistic} ({100*stylistic/total:.1f}%)")

    # ========== СОХРАНЕНИЕ СТАТИСТИКИ В JSON ==========
    stats_data = {
        'summary': {
            'total_markers': total_markers,
            'genuine_count': genuine_count,
            'genuine_percentage': round(100 * genuine_count / total_markers, 2),
            'stylistic_count': stylistic_count,
            'stylistic_percentage': round(100 * stylistic_count / total_markers, 2),
            'unclear_count': unclear_count,
            'unclear_percentage': round(100 * unclear_count / total_markers, 2)
        },
        'confidence_distribution': confidence_distribution,
        'marker_stats': marker_stats,
        'category_stats': category_stats,
        'generated_at': datetime.now().isoformat()
    }

    try:
        with open(STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        log_message(f"\nСтатистика сохранена: {STATS_FILE}")
    except Exception as e:
        log_message(f"Ошибка сохранения статистики: {e}")

    log_message("=" * 70)

def classify_all_corrections():
    """
    Основная функция: проходит по всем результатам и классифицирует каждый маркер.
    Поддерживает возобновление с чекпоинта через RESUME_FROM_CHECKPOINT.
    """

    log_message("=" * 70)
    log_message("КЛАССИФИКАЦИЯ САМОИСПРАВЛЕНИЙ")
    log_message("=" * 70)
    log_message(f"Модель-судья: {JUDGE_MODEL}")
    log_message(f"Входной файл: {RESULTS_FILE}")
    log_message(f"Выходной файл: {OUTPUT_FILE}")
    if RESUME_FROM_CHECKPOINT:
        log_message(f"Возобновление с чекпоинта: {RESUME_FROM_CHECKPOINT}")
    log_message("=" * 70 + "\n")

    # ========== ЗАГРУЗКА ЧЕКПОИНТА ==========
    if RESUME_FROM_CHECKPOINT:
        classified_results, processed_ids, processed_markers = load_checkpoint(RESUME_FROM_CHECKPOINT)
    else:
        classified_results, processed_ids, processed_markers = [], set(), 0

    # ========== ЗАГРУЗКА ДАННЫХ ==========
    log_message("Загружаю результаты экспериментов...")

    try:
        with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        log_message(f"ОШИБКА: Файл {RESULTS_FILE} не найден!")
        return False
    except json.JSONDecodeError as e:
        log_message(f"ОШИБКА: Невалидный JSON в {RESULTS_FILE}: {e}")
        return False

    # Извлекаем вопросы
    questions = data.get('questions', [])

    if not questions:
        log_message("ОШИБКА: В файле нет вопросов!")
        return False

    log_message(f"Загружено {len(questions)} вопросов")

    # Фильтруем уже обработанные вопросы
    questions_to_process = [
        q for idx, q in enumerate(questions)
        if q.get('id', idx + 1) not in processed_ids
    ]

    skipped_count = len(questions) - len(questions_to_process)
    if skipped_count > 0:
        log_message(f"Пропускаем {skipped_count} уже обработанных вопросов")

    log_message(f"Осталось обработать: {len(questions_to_process)} вопросов")

    # Подсчитываем общее количество маркеров (только для необработанных)
    remaining_markers = sum(
        sum(q.get('reflection_phrases', {}).values())
        for q in questions_to_process
    )
    log_message(f"Маркеров для классификации: {remaining_markers}\n")

    if not questions_to_process:
        log_message("Все вопросы уже обработаны!")
        generate_classification_stats(classified_results)
        return True

    # ========== ОБРАБОТКА ВОПРОСОВ ==========
    for q_idx, question_data in enumerate(tqdm(questions_to_process, desc="Обработка вопросов", unit="q")):
        question_id = question_data.get('id', q_idx + 1)
        question_text = question_data.get('question', '')
        category = question_data.get('category', 'unknown')
        thinking = question_data.get('full_thinking', '')
        final_answer = question_data.get('full_answer', '')
        reflection_phrases = question_data.get('reflection_phrases', {})

        # Пропускаем, если нет thinking
        if not thinking:
            log_message(f"Вопрос {question_id}: нет thinking текста, пропускаем")
            continue

        # Сохранение результатов по этому вопросу
        question_classifications = {
            'id': question_id,
            'category': category,
            'question_preview': question_text[:100] + '...' if len(question_text) > 100 else question_text,
            'total_markers': sum(reflection_phrases.values()),
            'markers': []
        }

        # Обрабатываем каждый тип маркера
        for marker, count in reflection_phrases.items():
            if count == 0:
                continue

            log_message(f"\nВопрос {question_id} | Маркер '{marker}' ({count} раз)")

            # Извлекаем все контексты для этого маркера
            contexts = extract_context(thinking, marker, window=50)

            if len(contexts) != count:
                log_message(f"Ожидалось {count} вхождений, найдено {len(contexts)}")

            marker_results = {
                'marker': marker,
                'expected_count': count,
                'found_count': len(contexts),
                'classifications': []
            }

            # Классифицируем каждое вхождение
            for ctx_idx, ctx in enumerate(contexts):
                log_message(f"Вхождение {ctx_idx+1}/{len(contexts)}...")

                classification = classify_correction(
                    context_before=ctx['before'],
                    marker=marker,
                    context_after=ctx['after'],
                    question=question_text
                )

                marker_results['classifications'].append({
                    'occurrence_index': ctx_idx + 1,
                    'position_in_tokens': ctx['position'],
                    'context_preview': ctx['full_context'][:150] + '...',
                    'classification': classification['classification'],
                    'confidence': classification['confidence'],
                    'reasoning': classification['reasoning']
                })

                processed_markers += 1

                # Небольшая пауза между запросами
                time.sleep(0.3)

            question_classifications['markers'].append(marker_results)

        classified_results.append(question_classifications)

        # Сохраняем промежуточный чекпоинт каждые 3 вопроса
        # Используем реальный номер вопроса для имени файла
        actual_processed_count = len(classified_results)
        if (q_idx + 1) % 3 == 0:
            checkpoint_file = f'checkpoint_classified_{actual_processed_count}.json'
            try:
                with open(checkpoint_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'processed_questions': actual_processed_count,
                        'total_questions': len(questions),
                        'processed_markers': processed_markers,
                        'total_markers': remaining_markers,
                        'questions': classified_results
                    }, f, ensure_ascii=False, indent=2)
                log_message(f"\nЧекпоинт сохранён: {checkpoint_file}")
            except Exception as e:
                log_message(f"Ошибка сохранения чекпоинта: {e}")

    # ========== СОХРАНЕНИЕ ФИНАЛЬНОГО РЕЗУЛЬТАТА ==========
    log_message(f"\n{'=' * 70}")
    log_message("Сохраняю результаты...")

    final_data = {
        'metadata': {
            'source_file': RESULTS_FILE,
            'judge_model': JUDGE_MODEL,
            'resumed_from_checkpoint': RESUME_FROM_CHECKPOINT,
            'total_questions_processed': len(classified_results),
            'total_markers_classified': processed_markers,
            'classification_date': datetime.now().isoformat()
        },
        'questions': classified_results
    }

    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        log_message(f"Результаты сохранены: {OUTPUT_FILE}")
    except Exception as e:
        log_message(f"Ошибка сохранения результатов: {e}")
        return False

    # ========== ГЕНЕРАЦИЯ СТАТИСТИКИ ==========
    generate_classification_stats(classified_results)

    log_message(f"\n{'=' * 70}")
    log_message("Классификация завершена!")
    log_message(f"{'=' * 70}\n")

    return True

if __name__ == "__main__":
    classify_all_corrections()

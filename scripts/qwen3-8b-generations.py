import json
import time
import sys
import argparse
from datetime import datetime
from ollama import chat

# ========== КОНФИГУРАЦИЯ ==========
MODEL = 'qwen3:8b'
DATASET_FILE = 'final_experiment_dataset.json'
RESULTS_FILE = 'experiment_results_continue.json'
CHECKPOINT_INTERVAL = 5

REFLECTION_PHRASES = [
    'wait', 'actually', 'let me reconsider', 'correction',
    'sorry', 'i meant', 'my mistake', 'on second thought',
    'i should re-examine', 'that seems wrong', 'i need to correct',
    'hang on', 'hold on', 'that\'s not right', 'i made an error'
]

def parse_arguments():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(description='Запуск эксперимента')
    parser.add_argument('--start', type=int, default=1, 
                       help='Номер вопроса, с которого начать (по умолчанию: 1)')
    parser.add_argument('--resume', action='store_true',
                       help='Продолжить с последнего сохранённого чекпоинта')
    parser.add_argument('--checkpoint', type=str,
                       help='Загрузить конкретный чекпоинт для продолжения')
    return parser.parse_args()

def log_message(msg):
    """Логирует сообщение в файл и выводит в консоль."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    full_msg = f"[{timestamp}] {msg}"
    
    print(full_msg)
    
    with open('experiment_continue.log', 'a', encoding='utf-8') as f:
        f.write(full_msg + '\n')

def load_dataset_and_checkpoint(start_from=1, checkpoint_file=None):
    """Загружает датасет и проверяет, какие вопросы уже обработаны."""
    
    # Загружаем датасет
    log_message(f"Загружаю датасет из {DATASET_FILE}...")
    try:
        with open(DATASET_FILE, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        log_message(f"Загружено {len(dataset)} вопросов")
    except Exception as e:
        log_message(f"Ошибка загрузки датасета: {e}")
        return None, None, None
    
    # Проверяем существующие результаты
    existing_results = []
    existing_ids = set()
    
    if checkpoint_file:
        # Загружаем из указанного чекпоинта
        try:
            with open(checkpoint_file, 'r', encoding='utf-8') as f:
                checkpoint_data = json.load(f)
            existing_results = checkpoint_data.get('questions', [])
            existing_ids = {item['id'] for item in existing_results if 'id' in item}
            log_message(f"Загружен чекпоинт {checkpoint_file}: {len(existing_results)} обработанных вопросов")
        except Exception as e:
            log_message(f"Не удалось загрузить чекпоинт: {e}")
    
    elif os.path.exists(RESULTS_FILE):
        # Загружаем из основного файла результатов
        try:
            with open(RESULTS_FILE, 'r', encoding='utf-8') as f:
                results_data = json.load(f)
            existing_results = results_data.get('questions', [])
            existing_ids = {item['id'] for item in existing_results if 'id' in item}
            log_message(f"Найдены предыдущие результаты: {len(existing_results)} обработанных вопросов")
        except Exception as e:
            log_message(f"Не удалось загрузить предыдущие результаты: {e}")
    
    # Определяем, какие вопросы нужно обработать
    questions_to_process = []
    
    for i, question in enumerate(dataset):
        question_id = question.get('id', i + 1)
        
        # Проверяем, был ли уже обработан этот вопрос
        if question_id in existing_ids:
            continue  # Пропускаем уже обработанные
        
        # Проверяем, нужно ли начинать с этого вопроса
        if question_id >= start_from:
            questions_to_process.append(question)
    
    log_message(f"Статистика:")
    log_message(f"   Всего вопросов в датасете: {len(dataset)}")
    log_message(f"   Уже обработано: {len(existing_results)}")
    log_message(f"   Осталось обработать: {len(questions_to_process)}")
    log_message(f"   Начинаю с ID: {questions_to_process[0]['id'] if questions_to_process else 'N/A'}")
    
    return dataset, existing_results, questions_to_process

def analyze_thinking(text):
    """Анализирует thinking-текст на наличие фраз саморефлексии."""
    if not text:
        return {}
    
    text_lower = text.lower()
    results = {}
    
    for phrase in REFLECTION_PHRASES:
        count = text_lower.count(phrase)
        if count > 0:
            results[phrase] = count
    
    return results

def ask_model(question, category, max_retries=3):
    """Запрашивает ответ у модели."""
    
    if category == 'math_trick':
        prompt = f"""Solve this mathematical problem carefully. 

Problem: {question}

After your reasoning, provide the final answer clearly."""
    
    elif category == 'conflicting':
        prompt = f"""Analyze the following scenario carefully. 

Scenario: {question}

Provide your analysis and final response."""
    
    else:  
        prompt = f"""Consider the following question carefully. 

Question: {question}

Provide your reasoned answer."""
    
    for attempt in range(max_retries):
        try:
            log_message(f"   Попытка {attempt+1}/{max_retries}...")
            
            response = chat(
                model=MODEL,
                messages=[{'role': 'user', 'content': prompt}],
                think=True,
                stream=False,
                options={
                    'temperature': 0.3,
                    'num_predict': 2500
                }
            )
            
            thinking_text = response.message.thinking or ""
            final_answer = response.message.content or ""
            
            return {
                'success': True,
                'thinking': thinking_text,
                'final_answer': final_answer,
                'thinking_length': len(thinking_text),
                'answer_length': len(final_answer)
            }
            
        except Exception as e:
            error_msg = str(e)
            if "model not found" in error_msg.lower():
                log_message(f"Модель {MODEL} не найдена!")
                log_message(f"   Загрузите её: ollama pull {MODEL}")
                break
            
            wait_time = 3 * (attempt + 1)
            log_message(f"   Ошибка: {error_msg[:80]}...")
            log_message(f"   Жду {wait_time} секунд...")
            time.sleep(wait_time)
    
    return {
        'success': False,
        'error': 'Все попытки не удались'
    }

def save_checkpoint(all_questions, checkpoint_num):
    """Сохраняет чекпоинт."""
    checkpoint_file = f"checkpoint_{checkpoint_num}.json"
    try:
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump({'questions': all_questions}, f, ensure_ascii=False, indent=2)
        log_message(f"  Чекпоинт сохранён: {checkpoint_file}")
        return True
    except Exception as e:
        log_message(f"Ошибка сохранения чекпоинта: {e}")
        return False

def save_final_results(all_questions):
    """Сохраняет финальные результаты."""
    final_data = {
        'model': MODEL,
        'dataset': DATASET_FILE,
        'total_questions': len(all_questions),
        'processed_time': datetime.now().isoformat(),
        'questions': all_questions
    }
    
    try:
        with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
            json.dump(final_data, f, ensure_ascii=False, indent=2)
        log_message(f"Финальные результаты сохранены: {RESULTS_FILE}")
        return True
    except Exception as e:
        log_message(f"Ошибка сохранения результатов: {e}")
        return False

# ========== ОСНОВНАЯ ФУНКЦИЯ ==========
def run_experiment_continue(start_from=1, checkpoint_file=None):
    """Запускает эксперимент с продолжением."""
    
    log_message("=" * 70)
    log_message("ЭКСПЕРИМЕНТ: ПРОДОЛЖЕНИЕ ОБРАБОТКИ")
    log_message("=" * 70)
    log_message(f"Модель: {MODEL}")
    log_message(f"Датасет: {DATASET_FILE}")
    log_message(f"Начинаю с вопроса: {start_from}")
    log_message(f"Время: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_message("=" * 70 + "\n")
    
    # Шаг 1: Загружаем датасет и существующие результаты
    dataset, existing_results, questions_to_process = load_dataset_and_checkpoint(
        start_from=start_from, 
        checkpoint_file=checkpoint_file
    )
    
    if not dataset or not questions_to_process:
        log_message("Нет вопросов для обработки")
        return False
    
    # Начинаем с существующих результатов
    all_questions = existing_results.copy()
    processed_count = len(existing_results)
    
    # Шаг 2: Обрабатываем оставшиеся вопросы
    total_to_process = len(questions_to_process)
    
    for idx, question_item in enumerate(questions_to_process):
        current_num = processed_count + idx + 1
        question_id = question_item.get('id', current_num)
        category = question_item.get('category', 'unknown')
        question_text = question_item.get('question', '')
        
        log_message(f"\n{'─' * 60}")
        log_message(f"ОБРАБОТКА ВОПРОСА {current_num} (ID: {question_id})")
        log_message(f"   Прогресс: {idx+1}/{total_to_process} оставшихся")
        log_message(f"   Категория: {category}")
        
        if len(question_text) > 100:
            log_message(f"   Текст: {question_text[:100]}...")
        else:
            log_message(f"   Текст: {question_text}")
        
        # Запрашиваем ответ у модели
        start_time = time.time()
        response = ask_model(question_text, category)
        processing_time = time.time() - start_time
        
        # Анализируем thinking-текст
        reflection_analysis = {}
        reflection_count = 0
        
        if response['success']:
            reflection_analysis = analyze_thinking(response['thinking'])
            reflection_count = sum(reflection_analysis.values())
        
        # Формируем результат
        result_entry = {
            'id': question_id,
            'category': category,
            'question': question_text,
            'success': response['success'],
            'processing_time_seconds': round(processing_time, 2),
            'reflection_count': reflection_count,
            'reflection_phrases': reflection_analysis,
            'thinking_preview': (response.get('thinking', '')[:150] + '...') if response.get('thinking') else '',
            'final_answer_preview': (response.get('final_answer', '')[:100] + '...') if response.get('final_answer') else '',
            'full_thinking': response.get('thinking', ''),
            'full_answer': response.get('final_answer', ''),
            'processed_at': datetime.now().isoformat()
        }
        
        if not response['success']:
            result_entry['error'] = response.get('error', 'Unknown error')
        
        # Добавляем к общим результатам
        all_questions.append(result_entry)
        
        # Логируем результат
        status = "УСПЕХ" if response['success'] else "ОШИБКА"
        reflection_info = f", найдено саморефлексий: {reflection_count}" if response['success'] else ""
        log_message(f"   {status}{reflection_info} ({processing_time:.1f} сек)")
        
        # Сохраняем чекпоинт каждые N вопросов
        if (idx + 1) % CHECKPOINT_INTERVAL == 0 or (idx + 1) == total_to_process:
            save_checkpoint(all_questions, current_num)
        
        # Пауза между запросами
        if idx < total_to_process - 1:
            time.sleep(2)
    
    # Шаг 3: Сохраняем финальные результаты
    save_final_results(all_questions)
    
    # Генерируем сводку
    generate_summary(all_questions)
    
    return True

def generate_summary(all_questions):
    """Генерирует сводку результатов."""
    successful = [q for q in all_questions if q.get('success', False)]
    failed = [q for q in all_questions if not q.get('success', False)]
    total_reflections = sum(q.get('reflection_count', 0) for q in successful)
    
    log_message(f"\n{'=' * 70}")
    log_message("ИТОГОВАЯ СТАТИСТИКА")
    log_message(f"{'=' * 70}")
    log_message(f"Всего обработано вопросов: {len(all_questions)}")
    log_message(f"Успешных: {len(successful)}")
    log_message(f"Ошибок: {len(failed)}")
    log_message(f"Всего случаев саморефлексии: {total_reflections}")
    
    if successful:
        avg_reflections = total_reflections / len(successful)
        log_message(f"Среднее саморефлексий на вопрос: {avg_reflections:.2f}")
    
    # Статистика по категориям
    categories = {}
    for question in successful:
        cat = question.get('category', 'unknown')
        if cat not in categories:
            categories[cat] = {'count': 0, 'reflections': 0}
        categories[cat]['count'] += 1
        categories[cat]['reflections'] += question.get('reflection_count', 0)
    
    log_message(f"ПО КАТЕГОРИЯМ:")
    for cat, stats in categories.items():
        if stats['count'] > 0:
            avg = stats['reflections'] / stats['count']
            log_message(f"  {cat}: {stats['count']} вопросов, {stats['reflections']} саморефлексий (avg: {avg:.2f})")
    
    # Сохраняем сводку в отдельный файл
    summary = {
        'total_processed': len(all_questions),
        'successful': len(successful),
        'failed': len(failed),
        'total_reflections': total_reflections,
        'avg_reflections': avg_reflections if successful else 0,
        'categories': categories,
        'completion_time': datetime.now().isoformat()
    }
    
    summary_file = 'experiment_summary_final.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    log_message(f"\nСводка сохранена: {summary_file}")
    log_message(f"{'=' * 70}")

# ========== ЗАПУСК ==========
if __name__ == "__main__":
    import os
    
    # Парсим аргументы
    args = parse_arguments()
    
    # Проверяем доступность Ollama
    try:
        from ollama import chat
        log_message("Библиотека Ollama доступна")
    except ImportError:
        print("Библиотека Ollama не установлена!")
        print("Установите: pip install ollama")
        sys.exit(1)
    
    # Определяем, с какого вопроса начинать
    start_question = args.start
    
    # Если указан флаг --resume, ищем последний чекпоинт
    checkpoint_to_load = None
    if args.resume:
        # Ищем последний чекпоинт
        checkpoint_files = [f for f in os.listdir('.') if f.startswith('checkpoint_') and f.endswith('.json')]
        if checkpoint_files:
            # Сортируем по номеру (самый последний)
            checkpoint_files.sort(key=lambda x: int(x.replace('checkpoint_', '').replace('.json', '')))
            checkpoint_to_load = checkpoint_files[-1]
            log_message(f"Найден последний чекпоинт: {checkpoint_to_load}")
            
            # Определяем, с какого вопроса продолжать (последний обработанный + 1)
            try:
                with open(checkpoint_to_load, 'r', encoding='utf-8') as f:
                    checkpoint_data = json.load(f)
                last_questions = checkpoint_data.get('questions', [])
                if last_questions:
                    last_id = max([q.get('id', 0) for q in last_questions])
                    start_question = last_id + 1
                    log_message(f"Последний обработанный ID: {last_id}, продолжаю с: {start_question}")
            except:
                pass
    
    # Если указан конкретный чекпоинт
    if args.checkpoint:
        checkpoint_to_load = args.checkpoint
        log_message(f"Использую указанный чекпоинт: {checkpoint_to_load}")
    
    # Запускаем эксперимент
    success = run_experiment_continue(
        start_from=start_question,
        checkpoint_file=checkpoint_to_load
    )
    
    sys.exit(0 if success else 1)
clear

[file, path] = uigetfile( ...
    {'*.txt','Текстовые файлы (*.txt)'; ...
     '*.*','Все файлы'}, ...
    'Выберите файлы для проверки', ...
    'MultiSelect', 'on');

if isequal(file,0)
    disp('Файлы не выбраны');
    return
end

% если выбран один файл — MATLAB вернёт char
if ischar(file)
    file = {file};
end

for i = 1:numel(file)
    fullpath = fullfile(path, file{i});
    fprintf("%s\n", fullpath);

    [authorConfidence, titleConfidence, specialtyConfidence, yearConfidence, cityConfidence] = processDocument(fullpath);

    docs(i).doc = file{i};
    docs(i).authorConfidence = authorConfidence;
    docs(i).titleConfidence = titleConfidence;
    docs(i).specialtyConfidence = specialtyConfidence;
    docs(i).yearConfidence = yearConfidence;
    docs(i).cityConfidence = cityConfidence;
end

figure(1)
clf(figure(1)) 

subplot(2,3,1)
hold on; grid on;
histogram([docs.authorConfidence], 10, 'Normalization', 'probability');
ylim([0 1])
title("Достоверность распознавания ФИО")

subplot(2,3,2)
hold on; grid on;
histogram([docs.titleConfidence], 10, 'Normalization', 'probability')
ylim([0 1])
title("Достоверность распознавания заголовка")

subplot(2,3,3)
hold on; grid on;
histogram([docs.specialtyConfidence], 10, 'Normalization', 'probability')
ylim([0 1])
title("Достоверность распознавания кода специальности")

subplot(2,3,4)
hold on; grid on;
histogram([docs.yearConfidence], 10, 'Normalization', 'probability')
ylim([0 1])
title("Достоверность распознавания года публикации")

subplot(2,3,5)
hold on; grid on;
histogram([docs.cityConfidence], 10, 'Normalization', 'probability')
ylim([0 1])
title("Достоверность распознавания города")

subplot(2,3,6)
hold on; grid on;
histogram([docs.cityConfidence docs.authorConfidence docs.titleConfidence docs.specialtyConfidence,docs.yearConfidence] ...
    , 10, 'Normalization', 'probability')
ylim([0 1])
title("Конкатенация достоверностей")

function [authorConfidence, titleConfidence, specialtyConfidence, yearConfidence, cityConfidence] =  processDocument(fullpath)

    text = fileread(fullpath);
    text = removeBarcodeNoise(text);

    author = extractAuthor(text);
    nErrors = 0;
    if isempty(author)
        fprintf(2,"%.1f | ", 0)
        confidence = 0;
    else
        [errors, nErrors] = parseYandexSpellResponse(yandexSpellCheckRequest(author));
        badChars = regexp(author, '[^а-яА-ЯёЁ\s\.-]', 'match');
        nErrors = nErrors + numel(badChars);      
        confidence = 1/(nErrors+1);        
        if (nErrors > 0)
            fprintf(2,"%.1f | ", confidence);
        else
            fprintf(1,"%.1f | ", confidence)
        end
    end
    fprintf("%s\n", author)   
    authorConfidence = confidence;
    title = extractTitle(text, author);   
    nErrors = 0;
    if isempty(title)
        fprintf(2,"%.1f | ", 0)
        confidence = 0;
    else
        [errors, nErrors] = parseYandexSpellResponse(yandexSpellCheckRequest(title));
        badChars = regexp(title, '[^а-яА-ЯёЁ\s\.-]', 'match');
        nErrors = nErrors + numel(badChars);     
        confidence = 1/(nErrors+1);
        if (nErrors > 0)
            fprintf(2,"%.1f | ", confidence)
        else
            fprintf(1,"%.1f | ", confidence)
        end
    end    
    fprintf("%s\n", title)    
    titleConfidence = confidence;    
    
    [specialty, nErrors] = extractSpecialty(text);
    if isempty(specialty)
        s = 0;
        fprintf(2,"%.1f | ", confidence)
    else   
        confidence = 1/(nErrors+1);
        if (nErrors > 0)
            fprintf(2,"%.1f | ", confidence)
        else
            fprintf(1,"%.1f | ", confidence)
        end
    end    
    fprintf("%s\n", specialty)    
    specialtyConfidence = confidence;
    
    year = extractYear(text);
    if isempty(year)
        confidence = 0;
        fprintf(2,"%.1f | ", confidence)
    else
        confidence = 1;
        fprintf(1,"%.1f | ", confidence)
    end 
    fprintf("%d\n", year)    
    yearConfidence = confidence;
    
    city = extractCity(text, year);
    if isempty(city)
        fprintf(2,"%.1f | ", 0)
        confidence = 0;
    else
        [errors, nErrors] = parseYandexSpellResponse(yandexSpellCheckRequest(city));
        badChars = regexp(city, '[^а-яА-ЯёЁ\s\.-]', 'match');
        nErrors = nErrors + numel(badChars);     
        confidence = 1/(nErrors+1);
        if (nErrors > 0)
            fprintf(2,"%.1f | ", confidence)
        else
            fprintf(1,"%.1f | ", confidence)
        end
    end      
    fprintf("%s\n", city)
    cityConfidence = confidence;
    
    
    
end


function author = extractAuthor(text)
    author = '';
    anchor = 'На\s+правах\s+рукописи';
    tokens = regexp(text, [anchor '([\s\S]{0,400})'], 'tokens', 'once');
    if isempty(tokens)
        return
    end
    window = tokens{1};
    lines = regexp(window, '\r?\n', 'split');
    fio_pattern = '^([А-ЯЁа-яё]{2,}(?:[ \t]+[А-ЯЁа-яё]{2,}){1,5})$';
    for i = 1:min(numel(lines), 10)  % ❗ ограничение
        line = strtrim(lines{i});
        if isempty(line) || strlength(line) > 80
            continue
        end
        if ~isempty(regexp(line, fio_pattern, 'once'))
            author = line;
            return
        end
    end
end

function title = extractTitle(text, authorLine)
    title = '';
    if isempty(authorLine)
        return
    end
    % ищем позицию ФИО
    idx = strfind(text, authorLine);
    if isempty(idx)
        return
    end
    % берём текст после ФИО
    tail = text(idx + length(authorLine):end);
    lines = regexp(tail, '\r?\n', 'split');
    buffer = {};
    for i = 1:min(5, numel(lines))
        line = strtrim(lines{i});
        if isempty(line)
            continue
        end
        % стоп-слова — признаки конца заголовка
        if contains(lower(line), {'специальност','автореферат','диссертац'})
            break
        end
        buffer{end+1} = line;
    end
    title = strjoin(buffer, ' ');
end

function [code, nErrors] = extractSpecialty(text)
    code = '';
    nErrors = 0;
    pattern = '(\d{2}[., ]\d{2}[., ]\d{2})';
    tokens = regexp(text, pattern, 'tokens', 'once');
    if isempty(tokens)
        return
    end
    raw = tokens{1};
    % извлекаем разделители
    separators = regexp(raw, '[^\d]', 'match');
    % считаем неправильные (всё, кроме точки)
    wrongSep = ~strcmp(separators, '.');
    nWrong = sum(wrongSep);
    nErrors = nWrong;
    % канонизируем код
    code = raw;
end

function year = extractYear(text)
    year = NaN;
    confidence = 0;
    pattern = '(?:\s|[-—])((19|20)\d{2})(\r?\n|$)';
    tokens = regexp(text, pattern, 'tokens');
    if isempty(tokens)
        return
    end
    % берём последний валидный год (обычно внизу титула)
    year = str2double(tokens{1}{1});
end

function city = extractCity(text, year)
    city = '';
    if isnan(year)
        return
    end
    lines = regexp(text, '\r?\n', 'split');
    for i = numel(lines):-1:1
        line = strtrim(lines{i});
        if isempty(regexp(line, num2str(year), 'once'))
            continue
        end
        % убираем год и разделители
        candidate = regexprep(line, ['[-—,\s]*' num2str(year)], '');
        candidate = strtrim(candidate);
        % город — кириллица, 1–3 слова
        if ~isempty(regexp(candidate, '^[А-ЯЁа-яё\s]{2,}$', 'once'))
            city = candidate;
            return
        end
    end
end

function xmlResponse = yandexSpellCheckRequest(text)
    % Отправляет текст в Яндекс.Спеллер и возвращает XML-ответ
    baseUrl = 'https://speller.yandex.net/services/spellservice/checkText';
    % URL-кодирование
    encodedText = char(java.net.URLEncoder.encode(text,'UTF-8'));
    query = sprintf('%s?text=%s', baseUrl, encodedText);
    options = weboptions( ...
        'Timeout', 10, ...
        'ContentType', 'text');
    xmlResponse = webread(query, options);
end


function [errors, nErrors] = parseYandexSpellResponse(xmlText)
    % Парсит XML-ответ Яндекс.Спеллера
    % Выводит:
    %   errors  - структура с полями word, suggestions, code, pos, len
    %   nErrors - количество ошибок в тексте

    errors = struct([]);
    nErrors = 0;

    if isempty(xmlText)
        return
    end

    % xmlread принимает файл или InputStream
    import java.io.*
    import org.xml.sax.InputSource

    is = InputSource();
    is.setCharacterStream(StringReader(xmlText));
    doc = xmlread(is);

    errorNodes = doc.getElementsByTagName('error');
    n = errorNodes.getLength();
    nErrors = n;

    for i = 1:n
        node = errorNodes.item(i-1);

        % атрибуты
        code = str2double(node.getAttribute('code'));
        pos  = str2double(node.getAttribute('pos'));
        len  = str2double(node.getAttribute('len'));

        % исходное слово
        wordNode = node.getElementsByTagName('word').item(0);
        word = char(wordNode.getTextContent());

        % подсказки
        sNodes = node.getElementsByTagName('s');
        suggestions = strings(1, sNodes.getLength());

        for j = 1:sNodes.getLength()
            suggestions(j) = string(sNodes.item(j-1).getTextContent());
        end

        errors(i).word = word;
        errors(i).suggestions = suggestions;
        errors(i).code = code;
        errors(i).pos = pos;
        errors(i).len = len;
    end
end

function isNoise = isBarcodeNoise(line)
    line = strtrim(line);
    if strlength(line) < 8
        isNoise = false;
        return
    end
    % 1) только I/1/l/|
    if ~isempty(regexp(line, '^[I1l|]{8,}$', 'once'))
        isNoise = true;
        return
    end
    % 2) слишком много одинаковых символов
    uniqueChars = unique(char(line));
    if numel(uniqueChars) <= 2 && strlength(line) >= 10
        isNoise = true;
        return
    end
    % 3) нет гласных
    if isempty(regexp(lower(line), '[аеёиоуыэюя]', 'once')) ...
       && strlength(line) >= 12
        isNoise = true;
        return
    end
    isNoise = false;
end

function cleanText = removeBarcodeNoise(text)

    lines = regexp(text, '\r?\n', 'split');
    keep = {};
    for i = 1:numel(lines)
        if ~isBarcodeNoise(lines{i})
            keep{end+1} = lines{i};
        end
    end
    cleanText = strjoin(keep, newline);
end
/*==============================================================*/
/* Helper functions - CSS
/*==============================================================*/
function cssSetElementToValues(element, propertyValueList) {
	if (propertyValueList.length % 2 != 0) throw "Error in cssSetElementToValues: " + propertyValueList + " must have an even-numbered length";
	for (let i = 0; i < propertyValueList.length; i += 2) {
		element.style.setProperty(propertyValueList[i], propertyValueList[i + 1]);
	}
}
function cssFindFirst(queryProperty) {
	return document.querySelector(queryProperty);
}
function cssFindAll(queryProperty) {
	return document.querySelectorAll(queryProperty);
}
function cssGetId(id) {
	const element = document.getElementById(id);
	assert(element != null, `Could not find id ${id}`);
	return element;
}
function cssSetId(id, property, value) {
	const element = cssGetId(id);
	assert(element != null, `Could not find id ${id}`);
    element.style.setProperty(property, value);
}
function cssSetIdToValues(id, propertyValueList) {
	const element = cssGetId(id);
	assert(element != null, `Could not find id ${id}`);
	cssSetElementToValues(element, propertyValueList);
}
function cssGetClass(className) {
	return document.getElementsByClassName(className);
}
function cssSetClass(className, property, value) {
	const elements = cssGetClass(className)
	for (let element of elements) {
		element.style.setProperty(property, value);
	}
}
function getForm(id) {
	const element = cssGetId(id);
	if (element.type == 'text' || element.type == 'range')
		return parseFloat(element.value);
	if (element.type == 'checkbox' || element.type == 'radio')
		return element.checked;
	if (element.type == 'color' || element.nodeName == 'SELECT')
		return element.value
	throw new Error(id);
}
function getActiveTab(part) {
	return cssGetClass(`tab-${part}-active`)[0];
}


/*==============================================================*/
/* Helper functions - Matrix operations
/*==============================================================*/
function matrixToString(arr) {
	let str = '[';
	for (let i = 0; i < arr.length; i++) {
		let suffix = (i == arr.length - 1) ? '' : ', ';
		if (Array.isArray(arr[i])) {
			str += matrixToString(arr[i]) + suffix;
		} else {
			str += arr[i] + suffix;
		}
	}
	return str + ']';
}
function elementWiseSet(matrix, operation) {
	for (let i = 0; i < matrix.length; i++) {
		for (let j = 0; j < matrix[i].length; j++) {
			matrix[i][j] = operation(matrix[i][j]);
		}
	}
}
function elementWiseIndexDo(matrix, operation) {
	for (let i = 0; i < matrix.length; i++) {
		for (let j = 0; j < matrix[i].length; j++) {
			operation(i, j);
		}
	}
}
function zeroMatrix(height, width) {
	matrix = Array.from({length: height}).fill(0);
	for (let i = 0; i < height; i++) {
		let row = Array.from({length: width}).fill(0);
		matrix[i] = row;
	}
	return matrix;
}
function emptyMatrix(height, width) {
	return Array.from({length: height}, () => Array(width));
}
function randomMatrix(height, width) {
	const data = [];
	for (let i = 0; i < height; i++) {
		data.push(Array.from({length: width}, () => 2 * Math.random() - 1));
	}
	return data;
}
function deepCopyMatrix(matrix) {
	const data = [];
	for (i = 0; i < matrix.length; i++) {
		const row = [];
		for (j = 0; j < matrix[i].length; j++) {
			row.push(matrix[i][j])
		}
		data.push(row);
	}
	return data;
}
function matrixTranspose(matrix) {
	const data = [];
	for (let j = 0; j < matrix[0].length; j++) {
		const row = [];
		for (let i = 0; i < matrix.length; i++) {
			row.push(matrix[i][j]);
		}
		data.push(row);
	}
	return data;
}
function matrixFromOperation(matrix, f) {
	const data = [];
	for (let i = 0; i < matrix.length; i++) {
		const row = [];
		for (let j = 0; j < matrix[i].length; j++) {
			row.push(f(matrix[i][j]));
		}
		data.push(row);
	}
	return data;
}
function matrixFromInplaceOperation(to, f, from) {
	elementWiseIndexDo(to, (i, j) => {
		to[i][j] = f(from[i][j])
	});
}
function flattenMatrix(matrix) {
	const flattened = [];
	elementWiseIndexDo(matrix, (i, j) => {
		flattened.push(matrix[i][j]);
	});
	return flattened;
}



/*==============================================================*/
/* Helper functions - General
/*==============================================================*/
function assert(condition, error) {
	if (!condition)
		throw new Error(error);
}
function bound(value, min, max) {
	assert(min <= max, `${min} > ${max}`);
	return Math.min(Math.max(value, min), max);
}
function roundToDecimal(x, n) {
	let pow = 10 ** n;
	return Math.round(x * pow) / pow;
}
function roundToFloat(x, n) {
	return Math.round(x * n) / n;
}
function arrayBounds1D(array) {
	let min = Infinity;
	let max = -Infinity;
	for (let value of array) {
		min = Math.min(value, min);
		max = Math.max(value, max);
	}
	return [min, max];
}
function euclideanDistance(x1, y1, x2, y2) {
	return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5;
}
function distanceToLine(x, y, x1, y1, x2, y2) {
	// https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
	return Math.abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / euclideanDistance(x1, y1, x2, y2);
}
function argsort(arr, i) {
	const indices = arr.map((value, index) => index);
    indices.sort((a, b) => arr[a][i] - arr[b][i]);
    return indices;
}


/*==============================================================*/
/* Toggles
/*==============================================================*/
function toggleTab(event) {
	const active = `tab-${CURR_PART}-active`;
	const curr = event.srcElement;
	const prev = cssGetClass(active)[0];
	prev.classList.remove(active);
	curr.classList.add(active);

	const prevId = prev.id.substring(prev.id.lastIndexOf('-') + 1, prev.id.length);
	const currId = curr.id.substring(curr.id.lastIndexOf('-') + 1, curr.id.length);
	cssSetId(`tab-body-${CURR_PART}-${prevId}`, 'display', 'none');
	cssSetId(`tab-body-${CURR_PART}-${currId}`, 'display', 'block');
	setCanvasSettings();
}
function toggleBrushMode(event) {
	const active = 'brush-mode-active';
	const curr = event.srcElement;
	const prev = cssGetClass(active)[0];
	prev.classList.remove(active);
	curr.classList.add(active);

	if (curr.id == 'brush-mode-elevation')
		cssSetClass('row-brush-elevation-only', 'display', 'table-row');
	else
		cssSetClass('row-brush-elevation-only', 'display', 'none');
}
function toggleGlobal(event) {
	const active = 'global-active';
	const curr = event.srcElement;
	const prev = cssGetClass(active)[0];
	prev.classList.remove(active);
	curr.classList.add(active);

	const id = curr.id;
	if (id == 'global-classification') {
		cssSetId('form-global-classification',		'display', 'block');
		cssSetId('month-slider-connector',			'display', 'none');
		cssSetId('month-slider',					'display', 'none');
		cssSetId('form-global-not-classification',	'display', 'none');
		cssSetId('form-global-temperature',			'display', 'none');
		cssSetId('form-global-precipitation',		'display', 'none');
	} else if (id == 'global-temperature') {
		cssSetId('form-global-classification',		'display', 'none');
		cssSetId('month-slider-connector',			'display', 'block');
		cssSetId('month-slider',					'display', 'grid');
		cssSetId('form-global-not-classification',	'display', 'block');
		cssSetId('form-global-temperature',			'display', 'block');
		cssSetId('form-global-precipitation',		'display', 'none');
	} else if (id == 'global-precipitation') {
		cssSetId('form-global-classification',		'display', 'none');
		cssSetId('month-slider-connector',			'display', 'block');
		cssSetId('month-slider',					'display', 'grid');
		cssSetId('form-global-not-classification',	'display', 'block');
		cssSetId('form-global-temperature',			'display', 'none');
		cssSetId('form-global-precipitation',		'display', 'block');
	} else {
		throw new Error('huh');
	}
	setCanvasSettings();
}
function setRadio() {
	setCanvasSettings();
}
function toggleStatistics(event) {
	const active = 'statistics-active';
	const curr = event.srcElement;
	const prev = cssGetClass(active)[0];
	prev.classList.remove(active);
	curr.classList.add(active);

	const id = curr.id;
	if (id == 'statistics-general') {
		cssSetId('table-statistics-general',	'display', 'table');
		cssSetId('latitude-canvas',				'display', 'none');
		cssSetId('elevation-canvas',			'display', 'none');
	} else if (id == 'statistics-latitude') {
		cssSetId('table-statistics-general',	'display', 'none');
		cssSetId('latitude-canvas',				'display', 'block');
		cssSetId('elevation-canvas',			'display', 'none');
	} else if (id == 'statistics-elevation') {
		cssSetId('table-statistics-general',	'display', 'none');
		cssSetId('latitude-canvas',				'display', 'none');
		cssSetId('elevation-canvas',			'display', 'block');
	} else {
		throw new Error('huh');
	}
}
function toggleSelectionMode(event) {
	const active = 'selection-mode-active';
	let curr = event.srcElement;
	const prev = cssGetClass(active)[0];
	prev.classList.remove(active);
	curr.classList.add(active);
}
function toggleSelectionShape(event) {
	const active = 'selection-shape-active';
	let curr = event.srcElement;
	const prev = cssGetClass(active)[0];
	prev.classList.remove(active);
	curr.classList.add(active);
}
function toggleCursorMode() {
	if (CURR_PART != 2)
		return;

	const tab = getActiveTab(CURR_PART).id;
	if (tab == 'tab-2-transform')
		return;
	if (CANVAS_MODE != 'display')
		CANVAS_MODE = 'display';
	else if (tab == 'tab-2-selection')
		CANVAS_MODE = 'selection';
	else
		CANVAS_MODE = 'brush';
}
function toggleLocalCanvas(event) {
	if (LOCAL_PIXEL == null)
		return;

	let id;
	if (event instanceof Event)
		id = event.srcElement.id;
	else
		id = (event) ? 'local-mini-canvas' : 'local-canvas';

	if (id == 'local-mini-canvas') {
		cssSetId('local-mini-canvas', 'display', 'none');
		const localCanvas = cssGetId('local-canvas');
		const localMiniCanvas = cssGetId('local-mini-canvas');
		localCanvas.getContext('2d').drawImage(localMiniCanvas, 0, 0);
		localCanvas.style.setProperty('display', 'block');
		localMiniCanvas.style.setProperty('display', 'none');
	} else {
		cssSetId('local-mini-canvas', 'display', 'block');
		cssSetId('local-canvas', 'display', 'none');
		cssSetId('map-canvas', 'display', 'block');
	}
}


/*==============================================================*/
/* Textnum
/*==============================================================*/
let TEMP;
let DISPLAY_DECIMAL = 2;
function beforeSetTextnum(event) {
	TEMP = event.srcElement.value;
}
function setTextnum(event) {
	const element = event.srcElement;
	if (element.value == TEMP)
		return;
	if (element.value == '' || isNaN(parseFloat(element.value))) {
		element.value = TEMP;
		return;
	}
	const suffix = FORM_VALUES[element.id][0];
	const min = parseFloat(element.getAttribute('data-min'));
	const max = parseFloat(element.getAttribute('data-max'));
	const newValue = bound(parseFloat(element.value), min, max);
	element.value = `${roundToDecimal(newValue, DISPLAY_DECIMAL)}${suffix}`;
	afterSetTextnum(event);
}
function initTextnum(event, delta) {
	const sign = (delta == null) ? -Math.sign(event.deltaY) : delta;
	const element = event.srcElement;
	const [suffix, deltaText] = FORM_VALUES[element.id];
	const min = parseFloat(element.getAttribute('data-min'));
	const max = parseFloat(element.getAttribute('data-max'));
	const newValue = bound(parseFloat(element.value) + sign * deltaText, min, max);
	element.value = `${roundToDecimal(newValue, DISPLAY_DECIMAL)}${suffix}`;

	event.preventDefault();
	event.stopPropagation();
	afterSetTextnum(event);
}
function afterSetTextnum(event) {
	const element = event.srcElement;
	const id = event.srcElement.id;
	const tab = getActiveTab(CURR_PART).id;
	if (CURR_PART == 1) {
		if (tab == 'tab-1-text') {
			const threshold = getForm('text-water-threshold');
			DATA_WATER = matrixFromOperation(DATA_ELEVATION_FINAL, x => x <= threshold);
		} else if (tab == 'tab-1-random') {
			if (id == 'x-axis-rotation' || id == 'y-axis-rotation' || id == 'z-axis-rotation') {
				generateEarthRandom(false);
			} else if (id == 'random-water-threshold') {
				const threshold = getForm('random-water-threshold');
				DATA_WATER = matrixFromOperation(DATA_ELEVATION_FINAL, x => x <= threshold);
			}
		}
		NO_NEW_MODIFICATIONS = false;
		drawData();
	} else if (CURR_PART == 2) {
		if (tab == 'tab-2-selection') {
			const textWidth = element.value.length * 1;
			const textSize = (textWidth > 3) ? '10px' : '12px';
			element.style.setProperty('font-size', textSize);
		} else {
			formatTextnum(element);

			const id = element.id;
			if (id.includes('elevation')) {
				PREVIEW = 'transform-elevation';
				let f;
				if (id == 'transform-multiply-elevation' || id == 'transform-add-elevation')
					f = transformSetElevation(true);
				else
					f = transformSetRange(true);
				matrixFromInplaceOperation(TRANSFORM_PREVIEW, f, DATA_FINAL);
			} else if (id == 'transform-threshold') {
				PREVIEW = 'transform-terrain';
				matrixFromInplaceOperation(TRANSFORM_PREVIEW, transformThreshold(true), DATA_FINAL);
			}
		}
		drawData();
	}
}
function formatTextnum(element) {
	const textWidth = element.value.length;
	const textSize = (textWidth > 7) ? '12px' : (textWidth > 5) ? '14px' : '16px';
	element.style.setProperty('font-size', textSize);

	// Textnum sliders
	let id = element.id;
	if (!(id in FORM_VALUES))
		id = element.id.substring(0, element.id.lastIndexOf('-'));

	// Exceptions
	const decimal = (element.id == 'transform-multiply-elevation') ? DISPLAY_DECIMAL * 2 : DISPLAY_DECIMAL;
	if (id == 'brush-threshold' && cssGetId('brush-threshold-textnum').value == 'OFF')
		return;

	const value = roundToDecimal(parseFloat(element.value), decimal);
	element.value = `${value}${FORM_VALUES[id][0]}`;
}
function afterSetTextnumSlider(slider, element) {
	if (slider.id == 'brush-threshold' && slider.value == slider.min)
		element.value = 'OFF';
	if (slider.id == 'brush-longitude')
		drawData();

	formatTextnum(element);
}
function setTextnumSlider(event) {
	const element = event.srcElement;
	if (element.value == TEMP)
		return;
	const slider = cssGetId(element.id.substring(0, element.id.lastIndexOf('-')));
	if (element.value.toLowerCase() == 'off') {
		element.value = 'OFF';
		slider.value = slider.min;
	}
	if (element.value == '' || isNaN(parseFloat(element.value))) {
		element.value = TEMP;
		return;
	}
	const suffix = FORM_VALUES[slider.id][0];
	const min = parseFloat(slider.min);
	const max = parseFloat(slider.max);
	const newValue = bound(parseFloat(element.value), min, max);
	element.value = `${newValue}${suffix}`;
	slider.value = newValue;

	afterSetTextnumSlider(slider, element);
}
function initTextnumSlider(event, delta) {
	const sign = (delta == null) ? -Math.sign(event.deltaY) : delta;
	const element = event.srcElement;
	const slider = cssGetId(element.id.substring(0, element.id.lastIndexOf('-')));
	const [suffix, deltaText, _] = FORM_VALUES[slider.id];
	const min = parseFloat(slider.min);
	const max = parseFloat(slider.max);
	const newValue = bound(parseFloat(slider.value) + sign * deltaText, min, max);
	element.value = `${newValue}${suffix}`;
	slider.value = newValue;

	afterSetTextnumSlider(slider, element);
	event.preventDefault();
	event.stopPropagation();
}
function setSlider(event) {
	const slider = event.srcElement;
	const element = cssGetId(`${slider.id}-textnum`);
	const suffix = FORM_VALUES[slider.id][0];
	element.value = `${slider.value}${suffix}`;

	afterSetTextnumSlider(slider, element);
}
function initSlider(event) {
	const sign = -Math.sign(event.deltaY);
	const slider = event.srcElement;
	const element = cssGetId(`${slider.id}-textnum`);
	const [suffix, _, deltaSlider] = FORM_VALUES[slider.id];
	const min = parseFloat(slider.min);
	const max = parseFloat(slider.max);
	const newValue = bound(parseFloat(slider.value) + sign * deltaSlider, min, max);
	element.value = `${newValue}${suffix}`;
	slider.value = newValue;

	afterSetTextnumSlider(slider, element);
	event.preventDefault();
	event.stopPropagation();
}


/*==============================================================*/
/* Inner textnum active
/*==============================================================*/
function imgButtonTextnumActive(event) {
	event.srcElement.classList.add('img-button-textnum-active');
}
function imgButtonTextnumInactive(event) {
	event.srcElement.classList.remove('img-button-textnum-active');
}

/*==============================================================*/
/* Checkbox
/*==============================================================*/
function setCheckbox(event) {
	const tab = getActiveTab(CURR_PART).id;
	const id = event.srcElement.id;
	if (CURR_PART == 1) {
		if (tab == 'tab-1-file') {
			drawData();
		} else if (tab == 'tab-1-text')
			inputText();
	} else if (CURR_PART == 3) {
		drawResult();
	}
}


/*==============================================================*/
/* Label Hover Captions
/*==============================================================*/
let CURR_LABEL = null;
function enterLabel(event) {
	const element = event.srcElement;
	const box = element.getBoundingClientRect();
	let id = element.getAttribute('for');
	if (id == null)
		id = element.id;
	const right = cssGetId(`tab-body-${CURR_PART}`).getBoundingClientRect().right;
	const caption = cssGetId(`${id}-caption`);

	cssSetElementToValues(caption, ['transition', '0s',
									'top', `${box.bottom}px`,
									'left', `${box.left}px`,
									'width', `${right - 15 - box.left}px`])
	CURR_LABEL = id;
	window.clearTimeout();
	const delay = (getActiveTab(CURR_PART).id == 'tab-2-selection') ? 500 : 300;
	setTimeout(() => {
		caption.style.setProperty('transition', '0.2s');
		if (CURR_LABEL == id)  // Prevents multiple labels from appearing when hovering between many quickly
			caption.style.setProperty('opacity', 0.9);
	}, delay);
}
function leaveLabel(event) {
	CURR_LABEL = null;
	cssSetClass('caption', 'opacity', 0);

	if (event.srcElement.id.includes('textnum'))
		imgButtonTextnumInactive(event);
}


/*==============================================================*/
/* Refreshing
/*==============================================================*/
function refreshInputForm() {
	if (DATA_ELEVATION_TYPE != 'npy' && DATA_ELEVATION_TYPE != 'image') {
		cssGetId('input-file').value = '';
		errorMessage('file', '');
	}
	if (DATA_ELEVATION_TYPE != 'text') {
		cssGetId('input-text').value = '';
		errorMessage('text', '');
	}

	// Form obstructions
	const type = (DATA_ELEVATION_TYPE == 'image') ? 'file' : DATA_ELEVATION_TYPE;
	for (let element of cssGetClass('form-obstruct-container')) {
		const parent = element.parentElement.id;
		const parentType = parent.substring(parent.lastIndexOf('-') + 1, parent.length);
		if (type == parentType) {
			cssSetElementToValues(element, ['opacity', '1', 'cursor', 'auto']);
			cssFindFirst(`#${parent} .form-obstruct`).style.setProperty('pointer-events', 'auto');
		} else {
			cssSetElementToValues(element, ['opacity', '0.5', 'cursor', 'not-allowed']);
			cssFindFirst(`#${parent} .form-obstruct`).style.setProperty('pointer-events', 'none');
		}
	}
}


/*==============================================================*/
/* Cities CSV
/*==============================================================*/
let CSV_HEADER;
let CSV;
function initCSV() {
	const data = CSV_RAW.split('\n');
	CSV_HEADER = data[0].split(',');
	CSV = [];

	for (let i = 1; i < data.length; i++) {
		const csvRow = [];
		const row = data[i].split(',');
		for (let j = 0; j < row.length; j++) {
			if (row[j].includes('"')) {
				let str = row[j];
				j += 1;
				while (!row[j].includes('"')) {
					str += row[j];
					j += 1;
				}
				csvRow.push(str.replace('"', ''));
			} else {
				csvRow.push(row[j]);
			}
		}
		// assert(csvRow.length == CSV_HEADER.length, `${csvRow.length}, ${CSV_HEADER.length}, [${csvRow}]`);
		CSV.push(csvRow);
	}
}


/*==============================================================*/
/* Initializing
/*==============================================================*/
const FORM_VALUES = {
	// suffix, deltaText, deltaSlider
	'file-min-elevation': ['m', 100],
	'file-max-elevation': ['m', 100],
	'file-max-elevation': ['m', 100],
	'text-water-threshold': ['m', 25],
	'x-axis-rotation': ['˚', 5],
	'y-axis-rotation': ['˚', 5],
	'z-axis-rotation': ['˚', 5],
	'scale-elevation': ['', 100],
	'offset-elevation': ['', 100],
	'random-water-threshold': ['m', 25],
	'brush-size': ['px', 2, 10],
	'brush-hardness': ['%', 2, 5],
	'brush-noise': ['m', 10, 50],
	'brush-distortion': ['%', 2, 5],
	'brush-threshold': ['m', 20, 100],
	'brush-elevation': ['m', 20, 100],
	'brush-longitude': ['˚', 2, 10],
	'transform-multiply-elevation': ['', 1],
	'transform-add-elevation': ['', 100],
	'transform-min-elevation': ['m', 100],
	'transform-max-elevation': ['m', 100],
	'transform-threshold': ['m', 20],
	'selection-above': ['', 100],
	'selection-below': ['', 100],
	'selection-min': ['', 100],
	'selection-max': ['', 100],
	'frame-length': ['ms', 10],
}
let DATA_ELEVATION_BACKUP = null;
let DATA_ELEVATION = null;
let DATA_ELEVATION_FINAL = null;
let DATA_WATER_BACKUP = null;
let DATA_WATER = null;
let DATA_DRAWING = null;
let DATA_FINAL = null;
let DATA_ELEVATION_TYPE;
let DATA_ELEVATION_BOUNDS;
let DATA_FINAL_BOUNDS;
let TRANSFORM_PREVIEW;
const MAX_ABSOLUTE_ELEVATION = 30000;
let CANVAS = null;
let CONTEXT = null;
let H, W;
function init() {
	CANVAS = cssGetId('map-canvas');
	CONTEXT = CANVAS.getContext("2d", {willReadFrequently: true, imageSmoothingEnabled: false});
	H = CANVAS.height;
	W = CANVAS.width;
	DATA_ELEVATION_FINAL = emptyMatrix(H, W);
	DATA_DRAWING = zeroMatrix(H, W);
	DATA_FINAL = emptyMatrix(H, W);
	RANDOM_MAP_2D = randomMatrix(H, W);
	TRANSFORM_PREVIEW = emptyMatrix(H, W)
	elementWiseIndexDo(DATA_FINAL, (y, x) => {
		SELECTION_ALL.add(hashIntCoord(x, y));
	});
	SELECTION = SELECTION_ALL;

	for (let element of cssGetClass('textnum')) {
		element.addEventListener("wheel", x => initTextnum(x, null));
	}
	for (let element of cssGetClass('textnum-slider')) {
		element.addEventListener("wheel", x => initTextnumSlider(x, null));
	}
	for (let element of cssGetClass('slider')) {
		element.addEventListener("wheel", initSlider);
	}
	for (let element of cssGetClass('textnum-save')) {
		element.addEventListener("wheel", x => initTextnum(x, null));
	}
	document.addEventListener('mousedown', (event) => {MOUSE_IS_DOWN = event.button == 0;});
	document.addEventListener('mouseup', (event) => {MOUSE_IS_DOWN = false;});
	document.addEventListener('mouseleave', (event) => {MOUSE_IS_DOWN = false;});
	initCSV();
}


/*==============================================================*/
/* Checks and errors
/*==============================================================*/
function errorMessage(type, message) {
	const id = (type == 'text') ? 'error-text' : 'error-file';
	const element = cssGetId(id);
	element.style.setProperty('color', 'rgb(255, 150, 140)');
	element.innerHTML = message;
	return null;
}
function warningMessage(type, message) {
	const id = (type == 'text') ? 'error-text' : 'error-file';
	const element = cssGetId(id);
	element.style.setProperty('color', 'rgb(222, 211, 140)');
	element.innerHTML = message;
	return null;
}
function checkAspectRatio(height, width) {
	if (Math.abs(width / height - 2) > 0.1)
		warningMessage(DATA_ELEVATION_TYPE, `Your input map's aspect ratio is ${roundToDecimal(width / height, 3)} ≠ 2 and will be stretched.`);
	else
		errorMessage(DATA_ELEVATION_TYPE, '');
}


/*==============================================================*/
/* Input file
/*==============================================================*/
function inputFile(event) {
	const fileHolder = cssGetId('input-file');
	if (fileHolder.files.length == 0)
		return;
	const file = fileHolder.files[0];

	if (file.name.endsWith('.npy')) {
		inputNPY(file);
		return;
		return inputData(elevation, water, min, max, height, width, 'npy');
	}

	const url = URL.createObjectURL(file);
	const img = new Image();
	img.src = url;
	img.onload = function() {
		CONTEXT.drawImage(img, 0, 0, W, H);
		const data = CONTEXT.getImageData(0, 0, W, H).data;
		const [elevation, water, min, max] = extractImageData(data);
		inputData(elevation, water, min, max, img.height, img.width, 'image');
		URL.revokeObjectURL(img.src);
	}
	img.onerror = function() {
		URL.revokeObjectURL(img.src);
		errorMessage('image', 'Something went wrong! Perhaps your filetype is not supported.');
	}
}
function inputNPY(file) {
	const reader = new FileReader();
	reader.onload = function(event) {
		try {
			const buffer = event.target.result;
			const ndarray = npy.frombuffer(buffer);

			if (ndarray.shape.length != 2)
				return errorMessage('npy', `Expected a 2D numpy array, got ${ndarray.shape}`);
			const [height, width] = ndarray.shape;
			const matrix = [];
			for (let i = 0; i < height; i++) {
				const row = [];
				for (let j = 0; j < width; j++) {
					row.push(ndarray.data[i * width + j]);
				}
				matrix.push(row);
			}
			const [elevation, min, max] = nearestNeighboursResample(matrix, H, W);
			const water = zeroMatrix(H, W);
			inputData(elevation, water, min, max, height, width, 'npy');
		} catch(Error) {
			errorMessage('npy', 'For some reason, the NPY could not be parsed.');
		}
	}
	reader.readAsArrayBuffer(file);
}
function extractImageData(clampedArray) {
	const data = [];
	const water = [];
	let min = Infinity;
	let max = -Infinity;
	let rowData = [];
	let rowWater = [];
	for (let i = 0; i < clampedArray.length; i += 4) {
		const r = clampedArray[i];
		const g = clampedArray[i + 1];
		const b = clampedArray[i + 2];
		const alpha = clampedArray[i + 3];
		const isWater = alpha != 255 || r != g || g != b || b != r;
		const value = (r + g + b) / (3 * 255);
		rowData.push(value);
		rowWater.push(isWater);
		min = Math.min(min, value);
		max = Math.max(max, value);

		if (rowData.length == W) {
			data.push(rowData);
			water.push(rowWater);
			rowData = [];
			rowWater = [];
		}
	}

	return [data, water, min, max];
}


/*==============================================================*/
/* Input text
/*==============================================================*/
function parseError(message) {
	return errorMessage('text', message);
}
function findTextNear(str, i) {
	const n = 5;
	const start = Math.max(i - n, 0);
	const end = Math.min(i + n + 1, str.length);
	const prefix = (start == i - n) ? '(...' : '([';
	const suffix = (end == i + n + 1) ? '...)' : '])';
	return prefix + str.substring(start, end) + suffix;
}
function inputText() {
	const element = cssGetId('input-text');
	const text = element.value;
	if (text == '')
		return parseError('');

	let data;
	if (text.endsWith(']]')) {
		if (!text.startsWith('[['))
			return parseError('Matrix ends with ]], so it should start with [[');
		data = parseMatrixTextV1(text)
	} else if (text.endsWith(']')) {
		if (!text.startsWith('['))
			return parseError('Matrix ends with ], so it should start with [');
		data = parseMatrixTextV2(text);
	} else {
		return parseError('Matrix should start/end with [...] or [[...]] brackets')
	}
	if (data == null)
		return;
	if (getForm('swap-rows-cols'))
		data = matrixTranspose(data);

	const height = data.length;
	const width = data[0].length;
	const threshold = getForm('text-water-threshold');
	const [elevation, min, max] = nearestNeighboursResample(data, H, W);
	const water = matrixFromOperation(elevation, x => x <= threshold);
	inputData(elevation, water, min, max, height, width, 'text');
}
function nearestNeighboursResample(matrix, newHeight, newWidth) {
	const oldHeight = matrix.length;
	const oldWidth = matrix[0].length;
	const data = [];

	let min = matrix[0][0];
	let max = matrix[0][0];
	for (let i = 0; i < newHeight; i++) {
		const rowData = [];
		for (let j = 0; j < newWidth; j++) {
			const row = Math.floor(i * oldHeight / newHeight);
			const column = Math.floor(j * oldWidth / newWidth);
			const value = matrix[row][column];

			rowData.push(value);
			min = Math.min(min, value);
			max = Math.max(max, value);
		}
		data.push(rowData);
	}
	return [data, min, max];
}
// [[a, b, c, ...], [a, b, c, ...], ...]
function parseMatrixTextV1(text) {
	let inMiddleOfRow = false;
	const isNumber = x => /^[\.-\d]$/.test(x);
	const data = [];
	let row = [];
	for (let i = 1; i < text.length - 1; i++) {
		if (inMiddleOfRow) {
			if (text[i] == '[')
				return parseError(`Extra [ detected at ${findTextNear(text, i)}`);

			if (text[i] == ']') {
				if (data.length > 0 && row.length != data[0].length)
					return parseError(`Row ${data.length}'s length is ${row.length}, expected ${data[0].length}`);
				data.push(row);
				row = [];
				inMiddleOfRow = false;

			} else if (isNumber(text[i])) {
				const numberStart = i;
				while (isNumber(text[i]) && i < text.length - 1) {
					i += 1;
				}
				if (i == text.length - 1)
					return parseError(`Expected ] at ${findTextNear(text, i)}`);
				const value = parseFloat(text.substring(numberStart, i));
				i -= 1;
				if (isNaN(value))
					return parseError(`Could not parse number at ${findTextNear(text, i)}`);
				row.push(value);
			}
		} else if (text[i] == '[') {
			inMiddleOfRow = true;
		} else if (isNumber(text[i])) {
			return parseError(`Unexpected number after end of row at ${findTextNear(text, i)}`);
		}
	}
	return data;
}
// [a, b, c, ... ; a, b, c, ..., ; ...]
function parseMatrixTextV2(text) {
	const isNumber = x => /^[\.-\d]$/.test(x);
	const data = [];
	let row = [];
	for (let i = 1; i < text.length; i++) {
		if (text[i] == ';') {
			if (data.length > 0 && row.length != data[0].length)
				return parseError(`Row ${data.length}'s length is ${row.length}, expected ${data[0].length}`);
			data.push(row);
			row = [];

		} else if (isNumber(text[i])) {
			const numberStart = i;
			while (isNumber(text[i]) && i < text.length - 1) {
				i += 1;
			}
			if (i == text.length)
				return parseError(`Unexpected number at ${findTextNear(text, i)}`);
			const value = parseFloat(text.substring(numberStart, i));
			i -= 1;
			if (isNaN(value))
				return parseError(`Could not parse number at ${findTextNear(text, i)}`);
			row.push(value);
		}
	}
	if (data.length > 0 && row.length != data[0].length)
		return parseError(`Row ${data.length}'s length is ${row.length}, expected ${data[0].length}`);
	data.push(row);
	return data;
}


/*==============================================================*/
/* Input Preset
/*==============================================================*/
function inputPreset(event) {
	const name = event.srcElement.value;
	const [elevation, min, max] = getPreset(name);
	const water = matrixFromOperation(elevation, x => x <= 0);
	inputData(elevation, water, min, max, H, W, 'preset');
}


/*==============================================================*/
/* Input Random
/*==============================================================*/
const RANDOM_MAP_3D = [];
function inputRandom() {
	generateEarthRandom(true);
}
function generateEarthRandom(reset) {
	if (reset)
		initEmpty3dMatrix();
	const elevation = sample3DMatrix();
	inputData(elevation, null, null, null, H, W, 'random');
}
function initEmpty3dMatrix() {
	const n = H;
	RANDOM_MAP_3D.length = 0;
	for (let i = 0; i < n; i++) {
		RANDOM_MAP_3D.push(emptyMatrix(n, n));
	}
}
function sample3DMatrix() {
	// https://en.wikipedia.org/wiki/Rotation_matrix
	const a = getForm('x-axis-rotation') * Math.PI / 180;
	const b = getForm('y-axis-rotation') * Math.PI / 180;
	const c = getForm('z-axis-rotation') * Math.PI / 180;
	const [sinA, cosA] = [Math.sin(a), Math.cos(a)];
	const [sinB, cosB] = [Math.sin(b), Math.cos(b)];
	const [sinC, cosC] = [Math.sin(c), Math.cos(c)];
	const A1 = cosB * cosC;
	const A2 = cosA * sinB * sinC - sinA * cosC;
	const A3 = cosA * sinB * cosC + sinA * sinC;
	const B1 = sinA * cosB;
	const B2 = sinA * sinB * sinC + cosA * cosC;
	const B3 = sinA * sinB * cosC - cosA * sinC;
	const C1 = -sinB;
	const C2 = cosB * sinC;
	const C3 = cosB * cosC;

	function rotation(x, y, z) {
		return [A1 * x + A2 * y + A3 * z,
				B1 * x + B2 * y + B3 * z,
				C1 * x + C2 * y + C3 * z];
	}

	// Skips and skip weights
	const skips = [60, 45, 30, 24, 18, 12, 9, 6, 4, 3, 2, 1];
	let skipSum = skips.reduce((a, b) => a + b, 0);
	let skipWeights = skips.map(x => (x / skipSum) ** 0.8);
	skipSum = skipWeights.reduce((a, b) => a + b, 0);
	skipWeights = skipWeights.map(x => x / skipSum);

	// Sampling
	const n = RANDOM_MAP_3D.length;
	const f = (x, r) => bound(Math.floor(x * r + r), 0, n - 1);
	const elevation = [];
	for (let i = 0; i < H; i++) {
		const row = []
		for (let j = 0; j < W; j++) {
			// I = latitude, J = longitude, r = radius
			const I = i * (Math.PI / 180);
			const J = j * (Math.PI / 180);
			const r = n / 2;
			const sinI = Math.sin(I);

			// Find 3D coordinate in cube
			const X = Math.cos(J) * sinI;
			const Y = Math.sin(J) * sinI;
			const Z = Math.cos(I);
			const [x, y, z] = rotation(X, Y, Z);
			const [_x, _y, _z] = [f(x, r), f(y, r), f(z, r)];

			// Find random noise value
			const value = smoothNoiseFunction3D(skips, skipWeights, _x, _y, _z);
			row.push(value * Math.abs(value));
		}
		elevation.push(row);
	}
	return elevation;
}
function smoothNoiseFunction3D(skips, skipWeights, x, y, z) {
	let sum = 0;
	for (let i = 0; i < skips.length; i++) {
		sum += interpolateNoise3D(RANDOM_MAP_3D, skips[i], x, y, z) * skipWeights[i];
	}
	return sum;
}
function interpolateNoise3D(randomMap3d, skip, x, y, z) {
	if (skip == 1)
		return matrixGet(randomMap3d, x, y, z);
	const n = randomMap3d.length;

	const prevX = Math.floor(x / skip) * skip;
	const nextX = (prevX + skip) % n;
	const prevY = Math.floor(y / skip) * skip;
	const nextY = (prevY + skip) % n;
	const prevZ = Math.floor(z / skip) * skip;
	// const nextZ = (prevZ + skip) % n;
	const nextZ = bound(prevZ + skip, 0, n - 1);

	const weightX = (x - prevX) / skip;
	const weightY = (y - prevY) / skip;
	const weightZ = (z - prevZ) / skip;

	const upperZ1 = matrixGet(randomMap3d, prevX, prevY, prevZ) * (1 - weightY) + matrixGet(randomMap3d, prevX, nextY, prevZ) * weightY;
	const lowerZ1 = matrixGet(randomMap3d, nextX, prevY, prevZ) * (1 - weightY) + matrixGet(randomMap3d, nextX, nextY, prevZ) * weightY;
	const upperZ2 = matrixGet(randomMap3d, prevX, prevY, nextZ) * (1 - weightY) + matrixGet(randomMap3d, prevX, nextY, nextZ) * weightY;
	const lowerZ2 = matrixGet(randomMap3d, nextX, prevY, nextZ) * (1 - weightY) + matrixGet(randomMap3d, nextX, nextY, nextZ) * weightY;
	const upperY = upperZ1 * (1 - weightX) + lowerZ1 * weightX;
	const lowerY = upperZ2 * (1 - weightX) + lowerZ2 * weightX;
	const result = upperY * (1 - weightZ) + lowerY * weightZ;
	return result;
}
function matrixGet(randomMap3d, x, y, z) {
	if (typeof randomMap3d[x][y][z] == 'undefined')
	 	randomMap3d[x][y][z] = 2 * Math.random() - 1;
	return randomMap3d[x][y][z];
}


/*==============================================================*/
/* Input
/*==============================================================*/
function inputData(elevation, water, min, max, height, width, type) {
	NO_NEW_MODIFICATIONS = false;
	DATA_ELEVATION_TYPE = type;
	DATA_ELEVATION_BOUNDS = [min, max];

	DATA_ELEVATION = elevation;
	DATA_ELEVATION_BACKUP = deepCopyMatrix(elevation);
	matrixFromInplaceOperation(DATA_DRAWING, x => 0, DATA_DRAWING);
	if (water != null) {
		DATA_WATER = water;
		DATA_WATER_BACKUP = deepCopyMatrix(water);
	}

	cssGetId('button-next').classList.remove('button-unselectable');
	refreshInputForm();
	checkAspectRatio(height, width);

	HISTORY.reset();
	drawData();
}
function drawData() {
	let preprocess;
	if (DATA_ELEVATION_TYPE == 'image') {
		const min = getForm('file-min-elevation');
		const max = getForm('file-max-elevation');
		const range = max - min;
		const invert = getForm('invert-image-shades');
		if (invert)
			preprocess = x => min + (1 - x) * range;
		else
			preprocess = x => min + x * range;

		if (getForm('min-elevation-as-water')) {
			const value = DATA_ELEVATION_BOUNDS[Number(invert)];
			DATA_WATER = matrixFromOperation(DATA_ELEVATION, x => Math.abs(x - value) <= 1e-10);
		} else {
			DATA_WATER = deepCopyMatrix(DATA_WATER_BACKUP);
		}
	} else if (DATA_ELEVATION_TYPE == 'random') {
		const scale = getForm('scale-elevation');
		const offset = getForm('offset-elevation');
		preprocess = x => offset + x * scale;

		const threshold = getForm('random-water-threshold');
		DATA_WATER = matrixFromOperation(DATA_ELEVATION, x => preprocess(x) <= threshold);
	} else {
		preprocess = x => x;
	}

	let min = Infinity;
	let max = -Infinity;
	elementWiseIndexDo(DATA_ELEVATION, (i, j) => {
		DATA_ELEVATION_FINAL[i][j] = preprocess(DATA_ELEVATION[i][j]);
		DATA_FINAL[i][j] = bound(DATA_ELEVATION_FINAL[i][j] + DATA_DRAWING[i][j], -MAX_ABSOLUTE_ELEVATION, MAX_ABSOLUTE_ELEVATION);
		min = Math.min(min, DATA_FINAL[i][j]);
		max = Math.max(max, DATA_FINAL[i][j]);
	});
	DATA_FINAL_BOUNDS = [min, max];
	drawCanvas(DATA_FINAL, DATA_WATER);
}
function drawCanvas(elevation, water) {
	if (water == null)
		water = zeroMatrix(H, W);

	const imageData = CONTEXT.getImageData(0, 0, W, H);
	const pixels = imageData.data;
	function colourPixels(r, g, b, offset, water, select) {
		if (select) {
			if (water) {
				pixels[offset] = 0;
				pixels[offset + 1] = 64;
				pixels[offset + 2] = 128;
			} else {
				pixels[offset] = r;
				pixels[offset + 1] = g;
				pixels[offset + 2] = b;
			}
		} else {
			if (water) {
				pixels[offset] = 128;
				pixels[offset + 1] = 32;
				pixels[offset + 2] = 64;
			} else {
				pixels[offset] = (r + 256) / 2;
				pixels[offset + 1] = g / 2;
				pixels[offset + 2] = b / 2;
			}
		}
		pixels[offset + 3] = 255;
	}

	// Previewing
	const selection = (SELECTION_TEMP == null) ? SELECTION : SELECTION_TEMP;
	let isWater = (i, j, hash) => DATA_WATER[i][j];
	let elevationAt = (i, j, hash) => DATA_FINAL[i][j];
	if (PREVIEW == 'terrain') {
		isWater = (i, j, hash) => TEMP.has(hash) ? TEMP.get(hash)[1]: DATA_WATER[i][j];
	} else if (PREVIEW == 'elevation') {
		elevationAt = (i, j, hash) => TEMP.has(hash) ? DATA_ELEVATION_FINAL[i][j] + TEMP.get(hash)[1] : DATA_FINAL[i][j];
	} else if (PREVIEW == 'transform-terrain') {
		isWater = (i, j, hash) => SELECTION.has(hash) ? TRANSFORM_PREVIEW[i][j] : DATA_WATER[i][j];
	} else if (PREVIEW == 'transform-elevation') {
		elevationAt = (i, j, hash) => SELECTION.has(hash) ? TRANSFORM_PREVIEW[i][j] : DATA_FINAL[i][j];
	} else if (PREVIEW == 'all-map') {
		elevationAt = (i, j, hash) => TEMP.has(hash) ? DATA_ELEVATION_FINAL[i][j] + TEMP.get(hash)[1] : DATA_FINAL[i][j];
		isWater = (i, j, hash) => TEMP.has(hash) ? TEMP.get(hash)[3] : DATA_WATER[i][j];
	}
	const longitudeOffset = getForm('brush-longitude-textnum');
	for (let i = 0; i < H; i++) {
		for (let j = 0; j < W; j++) { // J = actual matrix, j = drawing
			const offset = (i * W + j) * 4;
			const J = (j + longitudeOffset + W) % W;
			const hash = hashIntCoord(J, i);
			const value = elevationToPixelShade(elevationAt(i, J, hash));
			colourPixels(value, value, value, offset, isWater(i, J, hash), selection.has(hash));
		}
	}
	CONTEXT.putImageData(imageData, 0, 0);
}
function elevationToPixelShade(x) {
	return bound(Math.tanh(x / 3000) * 256, 0, 255);
}


/*==============================================================*/
/* Part transitioning
/*==============================================================*/
let CURR_PART = 1;
let NO_NEW_MODIFICATIONS;
function part1() {
	CURR_PART = 1;
	setCanvasSettings();
	drawData();

	cssSetId('part-1', 'margin-left', '0%');
	cssSetId('part-2', 'margin-left', '100%');
}
function part2() {
	if (DATA_ELEVATION == null)
		return;
	if (CURR_PART == 3)
		drawData();

	CURR_PART = 2;
	setCanvasSettings();

	cssSetId('colour-bar-block', 'opacity', '0');
	cssSetId('part-1', 'margin-left', '-100%');
	cssSetId('part-2', 'margin-left', '0%');
	cssSetId('part-3', 'margin-left', '100%');
}
function part3() {
	CURR_PART = 3;
	HISTORY.reset();
	setCanvasSettings();
	initCanvas('local-mini-canvas', 'Click a pixel on the canvas');
	updateStatisticsGeneral();
	plotStatisticsHistograms();

	cssSetId('colour-bar-block', 'opacity', '1');
	cssSetId('part-1', 'margin-left', '-200%');
	cssSetId('part-2', 'margin-left', '-100%');
	cssSetId('part-3', 'margin-left', '0%');
}


/*==============================================================*/
/* Cursor hovering on canvas
/*==============================================================*/
let CANVAS_MODE = 'display'; // display, brush, selection
let CANVAS_DISPLAY = 'elevation' // elevation, temperature, precipitation, koppen, trewartha
let MOUSE_IS_DOWN = false;
let CURSOR_X = null;
let CURSOR_Y = null;
let PIXEL_X = null;
let PIXEL_Y = null;

function setCanvasSettings() {
	const tab = getActiveTab(CURR_PART).id;
	PREVIEW = null;

	if (CURR_PART == 1) {
		CANVAS_MODE = 'display';
		CANVAS_DISPLAY = 'elevation';
	} else if (CURR_PART == 2) {
		CANVAS_DISPLAY = 'elevation';
		if (tab == 'tab-2-selection') {
			CANVAS_MODE = 'selection';
		} else if (tab == 'tab-2-brush') {
			CANVAS_MODE = 'brush';
		} else if (tab == 'tab-2-transform') {
			CANVAS_MODE = 'display';
			updateSelectionBounds(null, null);
		} else {
			throw new Error(tab);
		}
		resetLocalClimate();
		drawData();
	} else if (CURR_PART == 3) {
		CANVAS_MODE = 'display';
		if (tab != 'tab-3-local')
			toggleLocalCanvas(false);

		const plot = cssGetClass('global-active')[0].id;
		if (plot == 'global-classification') {
			if (getForm('global-classification-koppen'))
				CANVAS_DISPLAY = 'koppen';
			else if (getForm('global-classification-trewartha'))
				CANVAS_DISPLAY = 'trewartha';
			else
				throw new Error('huh');
		} else if (plot == 'global-temperature') {
			CANVAS_DISPLAY = 'temperature';
		} else if (plot == 'global-precipitation') {
			CANVAS_DISPLAY = 'precipitation';
		} else {
			throw new Error(plot);
		}
		drawResult();
	}
}
function trackMouse(event, canvas) {
	CURSOR_X = event.clientX;
	CURSOR_Y = event.clientY;
	const box = canvas.getBoundingClientRect();
	PIXEL_X = bound(Math.floor((CURSOR_X - box.left) * canvas.width / box.width), 0, canvas.width - 1);
	PIXEL_Y = bound(Math.floor((CURSOR_Y - box.top) * canvas.height / box.height), 0, canvas.height - 1);
}
function onCanvasMouseHover(event) {
	// Show/hide cursor
	if (DATA_ELEVATION == null)
		return cssSetId('map-canvas', 'cursor', 'auto');
	cssSetId('map-canvas', 'cursor', 'none');

	trackMouse(event, CANVAS);
	if (MOUSE_IS_DOWN) {
		if (CURR_PART == 3)
			displaySelector(true);
		return onCanvasMouseDown(event);
	} else if (CANVAS_MODE == 'brush') {
		return displayBrush();
	} else {
		return displaySelector(CANVAS_MODE == 'display');
	}
}
function hideCanvasCursor() {
	cssSetId('map-canvas-caption',		'opacity', '0');
	cssSetId('map-canvas-hover-pixel',	'opacity', '0');
	cssSetId('map-canvas-brush',		'opacity', '0');
	cssSetId('map-canvas-brush-looped', 'opacity', '0');
}
function onCanvasMouseExit() {
	hideCanvasCursor();
	if (MOUSE_IS_DOWN)
		onCanvasMouseUp(null);
}
function displaySelector(displayCaption) {
	const box = CANVAS.getBoundingClientRect();

	// Display hovering pixel
	const pixelLeft = box.left + PIXEL_X * box.width / W;
	const pixelTop = box.top + PIXEL_Y * box.height / H;
	const colour = (MOUSE_IS_DOWN && !displayCaption) ? 'rgb(150, 150, 150)' : 'white';
	cssSetIdToValues('map-canvas-hover-pixel',
					['opacity', '1',
					'border', `1px solid ${colour}`,
					'width', `${box.width / W}px`,
					'height', `${box.height / H}px`,
					'left', `${pixelLeft}px`,
					'top', `${pixelTop}px`]);

	if (displayCaption) {
		// Display hovering caption
		const caption = cssGetId('map-canvas-caption');
		const captionHeight = caption.getBoundingClientRect().height;
		cssSetElementToValues(caption, ['opacity', '1',
										'left', `${pixelLeft + 10}px`,
										'top', `${pixelTop - captionHeight / 2}px`])

		// Display hovering caption text
		const title = cssFindFirst('#map-canvas-caption h4');
		const decimals = 2;
		let value;
		if (PREVIEW == 'transform-elevation') {
			if (SELECTION.has(hashIntCoord(PIXEL_X, PIXEL_Y)))
				value = `${roundToDecimal(TRANSFORM_PREVIEW[PIXEL_Y][PIXEL_X], decimals)} m`;
			else
				value = `${DATA_FINAL[PIXEL_Y][PIXEL_X]} m`;
			return;
		}

		if (CANVAS_DISPLAY == 'elevation')
			value = `${roundToDecimal(DATA_FINAL[PIXEL_Y][PIXEL_X], decimals)} m`;
		else if (CLIMATE_NET['water'][PIXEL_Y][PIXEL_X])
			value = 'Water mass';
		else if (CANVAS_DISPLAY == 'temperature')
			value = roundToDecimal(CLIMATE_NET[`temp_${MONTHS['short'][CURR_MONTH]}`][PIXEL_Y][PIXEL_X], decimals) + '˚C';
		else if (CANVAS_DISPLAY == 'precipitation')
			value = roundToDecimal(CLIMATE_NET[`prec_${MONTHS['short'][CURR_MONTH]}`][PIXEL_Y][PIXEL_X], decimals) + ' mm';
		else if (CANVAS_DISPLAY == 'koppen')
			value = `${KOPPEN.labels[CLIMATE_NET['koppen'][PIXEL_Y][PIXEL_X]]}`;
		else if (CANVAS_DISPLAY == 'trewartha')
			value = `${TREWARTHA.labels[CLIMATE_NET['trewartha'][PIXEL_Y][PIXEL_X]]}`;
		else
			throw new Error("huh");
		title.innerText = value;

		const description = cssFindFirst('#map-canvas-caption p');
		const longitudeSymbol = (PIXEL_X > W / 2) ? 'E' : 'W';
		const latitudeSymbol = (PIXEL_Y > H / 2) ? 'S' : 'N';
		const longitude = Math.abs(PIXEL_X - W / 2);
		const latitude = Math.abs(PIXEL_Y - H / 2);
		description.innerText = `${longitude}° ${longitudeSymbol}, ${latitude}° ${latitudeSymbol}`;
	}
}

function displayBrush() {
	const box = CANVAS.getBoundingClientRect();
	const brushHeight = getForm('brush-size');
	const distortion = getDistortion();
	const brushWidth = brushHeight * distortion;

	// Display brush
	const colour = (MOUSE_IS_DOWN) ? 'rgb(150, 150, 150)' : 'white';
	function displayBrush(offset) {
		const id = (offset == 0) ? 'map-canvas-brush' : 'map-canvas-brush-looped';
		cssSetIdToValues(id, ['border',		`${brushHeight / 100}px solid ${colour}`,
							'height',		`${brushHeight}px`,
							'width',		`${brushWidth}px`,
							'margin-left',	`${CURSOR_X - box.left - (brushWidth / 2) + offset}px`,
							'margin-top',	`${CURSOR_Y - box.top - (brushHeight / 2)}px`,
							'opacity',		'1']);
	}
	displayBrush(0);
	const offset = (CURSOR_X - box.left > box.width / 2) ? -box.width : box.width;
	displayBrush(offset);
}
function getDistortion() {
	const distortionWeight = getForm('brush-distortion') / 100;
	if (distortionWeight == 0)
		return 1;

	const box = CANVAS.getBoundingClientRect();
	const canvasMiddle = (box.top + box.bottom) / 2;
	const latitude = bound(180 * Math.abs((CURSOR_Y - canvasMiddle) / box.height), 0, 90);
	return bound(distortionFunction(latitude), 0, 50);
}
function distortionFunction(latitude) {
	// Approximation
	// let distortionWeight = cssGetId('distortion').value / 100;
	// let maxDistortion = 8100 / Math.max(8100 - latitude ** 2, 0.0001);
	// return Math.sqrt(maxDistortion) ** (2.3584991 * distortionWeight);

	// Actual
	let radian = x => x * Math.PI / 180;
	let degree = x => x * 180 / Math.PI;
	let angle = radian(Math.abs(latitude - 90));
	let radius = 6371;
	let xCoord = (k, a) => Math.cos(k) * Math.sin(a) * radius;
	let yCoord = (k, a) => Math.sin(k) * Math.sin(a) * radius;
	let zCoord = (k, a) => Math.cos(a) * radius;

	let a1 = xCoord(0, angle);
	let a2 = yCoord(0, angle);
	let a3 = zCoord(0, angle);
	let b1 = xCoord(degree(1), angle);
	let b2 = yCoord(degree(1), angle);
	let b3 = zCoord(degree(1), angle);
	let circularDistance = radius * Math.acos(bound((a1 * b1 + a2 * b2 + a3 * b3) / (radius ** 2), 0, 1));
	return 4759.848949477168 / Math.max(circularDistance, 0.0001);
}


/*==============================================================*/
/* History
/*==============================================================*/
class HistoryAction {
	constructor(changes, type) {
		assert(type == 'elevation' || type == 'terrain' || type == 'all-matrix' || type == 'all-map' || type == 'selection', `Unexpected actino type ${type}`);
		this.changes = changes;
		this.type = type;
		this.prev = null;
		this.next = null;
	}
}
class History {
	constructor(capacity) {
		this.capacity = capacity;
		this.size = 0;
		this.currDepth = 0;
		this.start = null;
		this.end = null;
		this.curr = null;
	}
	reset() {
		this.size = 0;
		this.currDepth = 0;
		this.start = null;
		this.end = null;
		this.curr = null;
	}
	action(changes, type) {
		NO_NEW_MODIFICATIONS = false;
		const action = new HistoryAction(changes, type);
		this.executeAction(action, true);

		// Empty
		if (this.size == 0) {
			this.start = action;
			this.end = action;
			this.curr = action;
			this.size += 1;

		// Action after undo-ing everything
		} else if (this.curr == null) {
			this.start = action;
			this.end = action;
			this.curr = action;
			this.size = 1;
			this.currDepth = 0;

		// Action after undo
		} else if (this.curr != this.start) {
			this.curr.prev = action;
			action.next = this.curr;
			this.start = action;
			this.curr = action;
			this.size -= this.currDepth;
			this.currDepth = 0;

		// Full
		} else if (this.size == this.capacity) {
			action.next = this.start;
			this.start.prev = action;
			this.start = action;
			this.curr = action;

			delete this.end.changes;
			this.end.prev.next = null;
			this.end = this.end.prev;

		// Not full
		} else {
			action.next = this.start;
			this.start.prev = action;
			this.start = action;
			this.curr = action;
			this.size += 1;
		}
	}
	undo() {
		if (this.curr == null || CURR_PART != 2)
			return;
		this.executeAction(this.curr, false);
		this.curr = this.curr.next;
		this.currDepth += 1;
	}
	redo() {
		if (this.curr == this.start || CURR_PART != 2)
			return;
		this.curr = (this.curr == null) ? this.end : this.curr.prev;
		this.currDepth -= 1;
		this.executeAction(this.curr, true);
	}
	executeAction(action, redo) {
		const i = Number(redo);
		const type = action.type;
		const changes = action.changes;
		if (type == 'elevation') {
			changes.forEach(function(value, key, map) {
				const [x, y] = unhashIntCoord(key);
				DATA_DRAWING[y][x] = value[i];
			});
		} else if (type == 'terrain') {
			changes.forEach(function(value, key, map) {
				const [x, y] = unhashIntCoord(key);
				DATA_WATER[y][x] = value[i];
			});
		} else if (type == 'all-map') {
			changes.forEach(function(value, key, map) {
				const [x, y] = unhashIntCoord(key);
				DATA_DRAWING[y][x] = value[i];
				DATA_WATER[y][x] = value[2 + i];
			});
		} else if (type == 'all-matrix') {
			matrixFromInplaceOperation(DATA_DRAWING, x => x, changes[i]);
			matrixFromInplaceOperation(DATA_WATER, x => x, changes[2 + i]);
		} else if (type == 'selection') {
			SELECTION = changes[i];
			SELECTION_TEMP = null;
		} else {
			throw new Error("unknown action type");
		}
		drawData();
	}
}
const HISTORY = new History(15);

function onCanvasMouseUp(event) {
	MOUSE_IS_DOWN = false;
	if (event != null && event.button == 2) // Releasing a right-click should do nothing
		return;
	if (CANVAS_MODE == 'brush')
		brushUp();
	else if (CANVAS_MODE == 'selection')
		selectUp();
}
function onCanvasMouseDown(event) {
	if (DATA_ELEVATION == null)
		return;

	// Right-click
	const tab = getActiveTab(CURR_PART).id;
	if (event.button == 2)  {
		if (MOUSE_IS_DOWN)
			onCanvasMouseUp(null); // Starting a right-click should cancel left-click stuff
		toggleCursorMode();
		hideCanvasCursor();
		onCanvasMouseHover(event);
		return;
	}
	MOUSE_IS_DOWN = true; // event-listener isn't fast enough

	// Local canvas
	if (CURR_PART == 3 && !CLIMATE_NET['water'][PIXEL_Y][PIXEL_X]) {
		LOCAL_PIXEL = [PIXEL_X, PIXEL_Y];
		return plotLocalClimate(-100, -100);
	}

	// Left-click
	if (event.button == 0)
		hideCanvasCursor();

	if (tab == 'tab-2-brush')
		brushDown(event);
	else if (tab == 'tab-2-selection')
		selectDown(event);
	else
		displaySelector(true);
}


/*==============================================================*/
/* Selection buttons
/*==============================================================*/
function selectionButton(event) {
	const id = event.srcElement.id;
	let selection = null;
	if (id == 'selection-invert') {
		selection = selectionInvert();
	} else if (id == 'selection-expand') {
		selection = selectionExpand();
	} else if (id == 'selection-contract') {
		selection = selectionContract();
	} else if (id == 'selection-land') {
		selection = selectionTerrain(false);
	} else if (id == 'selection-water') {
		selection = selectionTerrain(true);
	} else if (id == 'selection-above-textnum') {
		selection = selectionAbove();
		imgButtonTextnumActive(event);
	} else if (id == 'selection-below-textnum') {
		selection = selectionBelow();
		imgButtonTextnumActive(event);
	} else if (id == 'selection-near-textnum') {
		selection = selectionNear();
		imgButtonTextnumActive(event);
	}
	if (selection != null)
		HISTORY.action([SELECTION, combineSelections(SELECTION, selection)], 'selection');
}
function selectionInvert() {
	const selection = new Set();
	elementWiseIndexDo(DATA_FINAL, (i, j) => {
		const hash = hashIntCoord(j, i);
		if (!SELECTION.has(hash))
			selection.add(hash);
	});
	return selection;
}
function selectionExpand() {
	const selection = new Set(SELECTION);
	SELECTION.forEach(function(value, key, map) {
		const [x, y] = unhashIntCoord(value);
		selection.add(hashIntCoord((x + 1) % W, y));
		selection.add(hashIntCoord((x - 1) % W, y));
		selection.add(hashIntCoord(x, Math.max(y - 1, 0)));
		selection.add(hashIntCoord(x, Math.min(y + 1, H - 1)));
	});
	return selection;
}
function selectionContract() {
	const selection = new Set(SELECTION);
	SELECTION.forEach(function(value, key, map) {
		const [x, y] = unhashIntCoord(value);
		if (!SELECTION.has(hashIntCoord((x + 1) % W, y))
				|| !SELECTION.has(hashIntCoord((x - 1 + W) % W, y))
				|| !SELECTION.has(hashIntCoord(x, Math.max(y - 1, 0)))
				|| !SELECTION.has(hashIntCoord(x, Math.min(y + 1, H - 1))))
			selection.delete(value);
	});
	return selection;
}
function selectionTerrain(isWater) {
	const selection = new Set();
	elementWiseIndexDo(DATA_WATER, (i, j) => {
		if (DATA_WATER[i][j] == isWater)
			selection.add(hashIntCoord(j, i));
	});
	return selection;
}
function selectionCondition(f) {
	const selection = new Set();
	elementWiseIndexDo(DATA_FINAL, (i, j) => {
		if (f(DATA_FINAL[i][j]))
			selection.add(hashIntCoord(j, i));
	});
	return selection;
}
function selectionAbove() {
	const threshold = getForm('selection-above');
	return selectionCondition(x => x >= threshold);
}
function selectionBelow() {
	const threshold = getForm('selection-below');
	return selectionCondition(x => x <= threshold);
}
function selectionNear() {
	const min = getForm('selection-min');
	const max = getForm('selection-max');
	return selectionCondition(x => (min <= x) && (x <= max));
}


/*==============================================================*/
/* Save states
/*==============================================================*/
const SAVE_STATE = [null, null];
function saveState() {
	if (SAVE_STATE[0] == null) {
		SAVE_STATE[0] = deepCopyMatrix(DATA_DRAWING);
		SAVE_STATE[1] = deepCopyMatrix(DATA_WATER);
	} else {
		matrixFromInplaceOperation(SAVE_STATE[0], x => x, DATA_DRAWING);
		matrixFromInplaceOperation(SAVE_STATE[1], x => x, DATA_WATER);
	}
}
function revertState() {
	if (SAVE_STATE[0] == null)
		return;
	PREVIEW = null;
	HISTORY.action([deepCopyMatrix(DATA_DRAWING), SAVE_STATE[0], deepCopyMatrix(DATA_WATER), SAVE_STATE[1]], 'all-matrix');
	if (getActiveTab(CURR_PART).id == 'tab-2-transform')
		updateSelectionBounds(null, null);
}
function resetAll() {
	PREVIEW = null;
	HISTORY.action([deepCopyMatrix(DATA_DRAWING), zeroMatrix(H, W), deepCopyMatrix(DATA_WATER), DATA_WATER_BACKUP], 'all-matrix');
	if (getActiveTab(CURR_PART).id == 'tab-2-transform')
		updateSelectionBounds(null, null);
}


/*==============================================================*/
/* Canvas selection
/*==============================================================*/
let SELECTION_ALL = new Set();
let SELECTION = new Set();
let SELECTION_TEMP = null;
let SELECT_START = null;
let SELECT_END = null;
let SELECTION_BOUNDS = null;

function hashIntCoord(x, y) {
	return ((x & 0xFFFF) << 16) | (y & 0xFFFF);
}
function unhashIntCoord(hash) {
	return [hash >> 16, hash & 0xFFFF];
}
function iterateEllipse(centerX, centerY, radiusX, radiusY, f) {
	const startX = Math.floor(centerX - radiusX);
	const endX = Math.ceil(centerX + radiusX);
	const upperY = (x) => Math.max(0, -radiusY * Math.sqrt(Math.max(1 - ((x - centerX) / radiusX) ** 2, 0)) + centerY);
	const lowerY = (x) => Math.min(H, radiusY * Math.sqrt(Math.max(1 - ((x - centerX) / radiusX) ** 2, 0)) + centerY);

	for (let x = startX; x <= endX; x++) {
		const startY = Math.floor(upperY(x));
		const endY = Math.ceil(lowerY(x));

		for (let y = startY; y <= endY; y++) {
			f(y, x);
		}
	}
}
function createSelection() {
	const shape = cssGetClass('selection-shape-active')[0].id;
	const selection = new Set();
	let [startX, startY] = SELECT_START;
	let [endX, endY] = SELECT_END;
	[startX, endX] = [Math.min(startX, endX), Math.max(startX, endX)];
	[startY, endY] = [Math.min(startY, endY), Math.max(startY, endY)];
	const longitudeOffset = getForm('brush-longitude');

	if (shape == 'selection-rectangle') {
		for (let i = startY; i <= endY; i++) {
			for (let j = startX; j <= endX; j++) {
				const J = (j + longitudeOffset + W) % W;
				selection.add(hashIntCoord(J, i));
			}
		}
	} else {
		const centerX = (startX + endX) / 2;
		const centerY = (startY + endY) / 2;
		const radiusX = Math.abs(endX - startX) / 2;
		const radiusY = Math.abs(endY - startY) / 2;
		iterateEllipse(centerX, centerY, radiusX, radiusY, (i, j) => {
			const J = (j + longitudeOffset + W) % W;
			selection.add(hashIntCoord(J, i));
		});
	}
	return selection;
}
function combineSelections(oldSelection, newSelection) {
	const mode = cssGetClass('selection-mode-active')[0].id;
	if (mode == 'selection-new') {
		return newSelection;
	} else if (mode == 'selection-union') {
		return oldSelection.union(newSelection);
	} else if (mode == 'selection-intersection') {
		return oldSelection.intersection(newSelection);
	} else if (mode == 'selection-subtraction') {
		return oldSelection.difference(newSelection);
	} else if (mode == 'selection-inversion') {
		return oldSelection.symmetricDifference(newSelection);
	} else {
		throw new Error("huh");
	}
}
function nullSelection() {
	const mode = cssGetClass('selection-mode-active')[0].id;
	if (mode == 'selection-new' || mode == 'selection-union') {
		return SELECTION_ALL;
	} else if (mode == 'selection-intersection' || mode == 'selection-subtraction' || mode == 'selection-inversion') {
		return new Set();
	} else {
		throw new Error("huh");
	}
}
function selectDown(event) {
	CANVAS_MODE = 'selection';
	if (SELECT_START == null) {
		SELECT_START = [PIXEL_X, PIXEL_Y];
	} else {
		SELECT_END = [PIXEL_X, PIXEL_Y];
		SELECTION_TEMP = combineSelections(SELECTION, createSelection());
		drawData();
	}
	displaySelector(false);
}
function selectUp(event) {
	if (SELECT_END == null)
		SELECTION_TEMP = nullSelection();
	HISTORY.action([SELECTION, SELECTION_TEMP], 'selection');

	SELECT_START = null;
	SELECT_END = null;
}
function updateSelectionBounds(min, max) {
	if (min == null) {
		min = Infinity;
		max = -Infinity;
		SELECTION.forEach(function(value, key, map) {
			const [x, y] = unhashIntCoord(key);
			const item = DATA_FINAL[y][x];
			min = Math.min(min, item);
			max = Math.max(max, item);
		});
	}
	SELECTION_BOUNDS = [min, max];
	const transformMinElevation = cssGetId('transform-min-elevation');
	const transformMaxElevation = cssGetId('transform-max-elevation');
	transformMinElevation.value = `${min}${FORM_VALUES['transform-min-elevation'][0]}`;
	transformMaxElevation.value = `${max}${FORM_VALUES['transform-max-elevation'][0]}`;
	formatTextnum(transformMinElevation);
	formatTextnum(transformMaxElevation);
}


/*==============================================================*/
/* Drawing on canvas
/*==============================================================*/
let PREVIEW = null;
let RANDOM_MAP_2D = null;

function brushUp() {
	const borderWidth = getForm('brush-size') / 100;
	cssSetId('map-canvas-brush', 'border', `${borderWidth}px solid white`);
	cssSetId('map-canvas-brush-looped', 'border', `${borderWidth}px solid white`);
	if (getForm('brush-noise') > 0)
		RANDOM_MAP_2D = randomMatrix(H, W);

	if (PREVIEW == 'all-map' || PREVIEW == 'elevation' || PREVIEW == 'terrain')
		HISTORY.action(TEMP, PREVIEW);
	else if (SELECTION_TEMP != null)
		HISTORY.action([SELECTION, SELECTION_TEMP], 'selection');

	PREVIEW = null;
	TEMP = null;
	SELECTION_TEMP = null;
}
function brushDown(event) {
	CANVAS_MODE = 'brush';
	displayBrush(event);

	// Brush click effect
	const borderWidth = getForm('brush-size') / 100;
	cssSetId('map-canvas-brush', 'border', `${borderWidth}px solid rgb(150, 150, 150)`);
	cssSetId('map-canvas-brush-looped', 'border', `${borderWidth}px solid rgb(150, 150, 150)`);

	// Find brush center + radius
	const box = CANVAS.getBoundingClientRect();
	const ratioWidth = W / box.width;
	const ratioHeight = H / box.height;
	const brush = cssGetId('map-canvas-brush').getBoundingClientRect();
	const centerX = (0.5 * (brush.left + brush.right) - box.left) * ratioWidth;
	const centerY = (0.5 * (brush.top + brush.bottom) - box.top) * ratioHeight;
	const radiusX = Math.min(W * ratioWidth, 0.5 * (brush.right - brush.left) * ratioWidth);
	const radiusY = 0.5 * (brush.bottom - brush.top) * ratioHeight;

	const distortion = getDistortion();
	const longitudeOffset = getForm('brush-longitude');
	const mode = cssGetClass('brush-mode-active')[0].id;

	if (mode == 'brush-mode-elevation') {
		if (!(TEMP instanceof Map))
			TEMP = new Map();

		const threshold = getForm('brush-threshold');
		let update;
		if (parseFloat(cssGetId('brush-threshold').min) == threshold) {
			PREVIEW = 'elevation';
			update = (hash, x, y, value) => {
				TEMP.set(hash, [DATA_DRAWING[y][x], value]);
			};
		} else {
			PREVIEW = 'all-map';
			update = (hash, x, y, value) => {
				TEMP.set(hash, [DATA_DRAWING[y][x], value, DATA_WATER[y][x], DATA_ELEVATION_FINAL[y][x] + value <= threshold]);
			};
		}

		const hardness = getForm('brush-hardness') / 100;
		const noise = getForm('brush-noise');
		const deltaElevation = getForm('brush-elevation');
		const noiseSkips = [180, 90, 60, 30, 15, 12, 10, 6, 3, 2, 1];

		iterateEllipse(centerX, centerY, radiusX, radiusY, (y, x) => {
			const X = (x + longitudeOffset + W) % W; // X = actual matrix, x = drawing
			const hash = hashIntCoord(X, y);

			if (SELECTION.has(hash)) {
				const distance = pixelBrushDistance(x, y, centerX, centerY, distortion);
				const hardnessValue = getHardnessValue(distance, radiusY, hardness);
				const noiseValue = getNoiseValue(X, y, noise, noiseSkips);
				const oldValue = (TEMP.has(hash)) ? TEMP.get(hash)[1] : DATA_DRAWING[y][X];
				const newValue = oldValue + hardnessValue * deltaElevation + noiseValue;
				const lower = -MAX_ABSOLUTE_ELEVATION - DATA_ELEVATION_FINAL[y][X];
				const upper = MAX_ABSOLUTE_ELEVATION - DATA_ELEVATION_FINAL[y][X];
				const finalValue = bound(newValue, lower, upper);
				update(hash, X, y, finalValue);
			}
		});
	} else if (mode == 'brush-mode-selection') {
		if (SELECTION_TEMP == null)
			SELECTION_TEMP = new Set();

		iterateEllipse(centerX, centerY, radiusX, radiusY, (y, x) => {
			const X = (x + longitudeOffset + W) % W; // X = actual matrix, x = drawing
			const hash = hashIntCoord(X, y);
			SELECTION_TEMP.add(hash);
		});

	} else if (mode == 'brush-mode-land' || mode == 'brush-mode-water') {
		if (!(TEMP instanceof Map))
			TEMP = new Map();
		PREVIEW = 'terrain';
		const terrain = mode == 'brush-mode-water';

		iterateEllipse(centerX, centerY, radiusX, radiusY, (y, x) => {
			const X = (x + longitudeOffset + W) % W; // X = actual matrix, x = drawing
			const hash = hashIntCoord(X, y);
			if (SELECTION.has(hash)) {
				TEMP.set(hash, [DATA_WATER[y][X], terrain]);
			}
		});
	} else {
		throw new Error(mode);
	}
	drawData();
}
function pixelBrushDistance(x, y, centerX, centerY, distortion) {
	const _x = centerX - ((centerX - x) / distortion);
	return Math.sqrt((_x - centerX) ** 2 + (y - centerY) ** 2);
}
function getHardnessValue(distance, radiusY, hardness) {
	if (hardness == 1)
		return 1;
	return bound((1 - bound((distance / radiusY), 0, 1) ** 2), 0, 1) ** bound(2 * (1 - hardness), 0, 2);
}
function getNoiseValue(j, i, noise, skips) {
	let sum = 0;
	for (let skip of skips) {
		sum += interpolateNoise(skip, i, j);
	}
	return noise * sum / skips.length;
}
function interpolateNoise(skip, i, j) {
	if (skip == 1)
		return RANDOM_MAP_2D[i][j];

	const prevI = Math.floor(i / skip) * skip;
	const nextI = (prevI + skip) % H;
	const prevJ = Math.floor(j / skip) * skip;
	const nextJ = (prevJ + skip) % W;

	const weightI = (i - prevI) / skip;
	const weightJ = (j - prevJ) / skip;

	const upper = RANDOM_MAP_2D[prevI][prevJ] * (1 - weightJ) + RANDOM_MAP_2D[prevI][nextJ] * weightJ;
	const lower = RANDOM_MAP_2D[nextI][prevJ] * (1 - weightJ) + RANDOM_MAP_2D[nextI][nextJ] * weightJ;
	return upper * (1 - weightI) + lower * weightI;
}


/*==============================================================*/
/* Transform buttons
/*==============================================================*/
function transformButton(event) {
	const id = event.srcElement.id;
	let changes, type;
	let min = null;
	let max = null;
	if (id == 'transform-flip-lr') {
		[changes, type] = transformFlip(false);
	} else if (id == 'transform-flip-ud') {
		[changes, type] = transformFlip(true);
	} else if (id == 'transform-land') {
		[changes, type] = transformTerrain(false);
	} else if (id == 'transform-water') {
		[changes, type] = transformTerrain(true);
	} else if (id == 'transform-set-elevation-confirm') {
		[changes, type] = transformSetElevation(false);
	} else if (id == 'transform-set-range-confirm') {
		[changes, type] = transformSetRange(false);
		min = getForm('transform-min-elevation');
		max = getForm('transform-max-elevation');
	} else if (id == 'transform-threshold-confirm') {
		[changes, type] = transformThreshold(false);
	} else {
		throw new Error(id);
	}
	HISTORY.action(changes, type);

	cssGetId('transform-multiply-elevation').value = '1';
	cssGetId('transform-add-elevation').value = '0';
	PREVIEW = null;
	updateSelectionBounds(min, max);
}
function transformFlip(vertical) {
	const drawing = emptyMatrix(H, W);
	const water = emptyMatrix(H, W);

	const fx = (vertical) ? x => x : x => W - 1 - x;
	const fy = (vertical) ? y => H - 1 - y : y => y;
	elementWiseIndexDo(DATA_FINAL, (i, j) => {
		const I = fy(i);
		const J = fx(j);
		drawing[i][j] = DATA_FINAL[I][J] - DATA_ELEVATION_FINAL[i][j];
		water[i][j] = DATA_WATER[I][J];
	});
	return [[deepCopyMatrix(DATA_DRAWING), drawing, deepCopyMatrix(DATA_WATER), water], 'all-matrix'];
}
function transformTerrain(water) {
	const changes = new Map();
	SELECTION.forEach(function(value, key, map) {
		const [x, y] = unhashIntCoord(key);
		const prev = DATA_WATER[y][x];
		changes.set(key, [prev, water]);
	});
	return [changes, 'terrain'];
}
function transformGeneric(f) {
	const changes = new Map();
	SELECTION.forEach(function(value, key, map) {
		const [x, y] = unhashIntCoord(key);
		const prev = DATA_DRAWING[y][x];
		const curr = f(DATA_FINAL[y][x]) - DATA_ELEVATION_FINAL[y][x];
		changes.set(key, [prev, curr]);
	});
	return [changes, 'elevation'];
}
function transformSetElevation(preview) {
	const multiply = getForm('transform-multiply-elevation');
	const add = getForm('transform-add-elevation');
	const f = x => x * multiply + add;
	if (preview)
		return f;
	return transformGeneric(f);
}
function transformSetRange(preview) {
	const [oldMin, oldMax] = SELECTION_BOUNDS;
	const oldRange = oldMax - oldMin;
	const newMin = getForm('transform-min-elevation');
	const newMax = getForm('transform-max-elevation');
	const newRange = newMax - newMin;
	const f = x => (x - oldMin) / oldRange * newRange + newMin;
	if (preview)
		return f;
	return transformGeneric(f);
}
function transformThreshold(preview) {
	const threshold = getForm('transform-threshold');
	const changes = new Map();
	if (preview)
		return x => x <= threshold;
	SELECTION.forEach(function(value, key, map) {
		const [x, y] = unhashIntCoord(key);
		const prev = DATA_WATER[y][x];
		changes.set(key, [prev, DATA_FINAL[y][x] <= threshold]);
	});
	return [changes, 'terrain'];
}


/*==============================================================*/
/* Keyboard shortcuts
/*==============================================================*/
document.addEventListener('keydown', function(event) {
	// Undo-redo
	if (event.ctrlKey && CURR_PART == 2) {
		if (event.key == 'z')
			return HISTORY.undo();
		else if (event.key == 'y')
			return HISTORY.redo();
	}
	// Textnum
	if (event.key == 'ArrowUp' || event.key == 'ArrowDown') {
		const element = document.activeElement;
		const id = element.id;
		const delta = (event.key == 'ArrowUp') ? 1 : -1;
		if (element.classList.contains('textnum-slider'))
			initTextnumSlider(event, delta);
		else if (element.classList.contains('textnum'))
			initTextnum(event, delta);
		return;
	}
	// Month slider
	if ((event.key == 'ArrowLeft' || event.key == 'ArrowRight')) {
		const active = document.activeElement;
		if (active.id != 'month-slider')
			return;
		const delta = (event.key == 'ArrowRight') ? 1 : -1;
		setMonthSlider((CURR_MONTH + delta + 12) % 12);
	}
});


/*==============================================================*/
/* Submit form
/*==============================================================*/
let CLIMATE_NET;
let PREDICT_BUSY = false;
function togglePredictBusy() {
	const element = cssGetId('button-predict');
	if (!PREDICT_BUSY) {
		element.value = 'Loading...';
		element.classList.add('button-unselectable');
	} else {
		element.value = 'Predict Climate';
		element.classList.remove('button-unselectable');
	}
	PREDICT_BUSY = !PREDICT_BUSY;
}

function predictClimate() {
	if (PREDICT_BUSY)
		return;
	if (NO_NEW_MODIFICATIONS)
		return part3();
	togglePredictBusy();

	// No backend
	/*
	part3();
	togglePredictBusy();
	*/

	// Backend
	$.ajax({
		url: '/',
		type: 'POST',
		contentType: 'application/json; chartset=utf-8',
		dataType: 'text json',
		data: JSON.stringify({'elevation': DATA_FINAL, 'water': DATA_WATER}),
		success: function(response) {
			// console.log(response);
			CLIMATE_NET = {
				temp_jan: response['temp'][0],
				temp_feb: response['temp'][1],
				temp_mar: response['temp'][2],
				temp_apr: response['temp'][3],
				temp_may: response['temp'][4],
				temp_jun: response['temp'][5],
				temp_jul: response['temp'][6],
				temp_aug: response['temp'][7],
				temp_sep: response['temp'][8],
				temp_oct: response['temp'][9],
				temp_nov: response['temp'][10],
				temp_dec: response['temp'][11],
				prec_jan: response['prec'][0],
				prec_feb: response['prec'][1],
				prec_mar: response['prec'][2],
				prec_apr: response['prec'][3],
				prec_may: response['prec'][4],
				prec_jun: response['prec'][5],
				prec_jul: response['prec'][6],
				prec_aug: response['prec'][7],
				prec_sep: response['prec'][8],
				prec_oct: response['prec'][9],
				prec_nov: response['prec'][10],
				prec_dec: response['prec'][11],
				water: response['water'],
				temp_range: response['temp_range'],
				prec_range: response['prec_range'],
				koppen: response['koppen'],
				trewartha: response['trewartha'],
				lowest_point: response['statistics'][0],
				highest_point: response['statistics'][1],
				farthest_land: response['statistics'][2],
				farthest_water: response['statistics'][3]
			};
			part3();
			togglePredictBusy();
			NO_NEW_MODIFICATIONS = true;
		},
		error: function(error) {
			console.log("Uh oh, something went wrong...");
			console.log(error);
			togglePredictBusy();
		}
	});
}


/*==============================================================*/
/* Month slider
/*==============================================================*/
let CURR_MONTH = 0;
function setMonthSlider(i) {
	const classCircleActive = 'month-slider-circle-active';
	const classTextActive = 'month-slider-text-active';
	const circleActive = cssGetClass(classCircleActive)[0];
	const textActive = cssGetClass(classTextActive)[0];
	const circle = cssGetId(`month-slider-circle-${i}`);
	const text = cssGetId(`month-slider-text-${i}`);
	CURR_MONTH = i;

	circleActive.classList.remove(classCircleActive);
	circle.classList.add(classCircleActive);
	textActive.classList.remove(classTextActive);
	text.classList.add(classTextActive);
	drawResult();
}
function wheelMonthSlider(event) {
	const delta = (event.deltaY > 0) ? 1 : -1;
	setMonthSlider((CURR_MONTH + delta + 12) % 12);
}
function scrollMonthSlider(event) {
	if (MOUSE_IS_DOWN) // Drag
		clickMonthSlider(event);
}
function clickMonthSlider(event) {
	let element = event.srcElement;
	if (element.id == 'month-slider')
		return;
	if (element.nodeName == 'SPAN')
		element = element.parentElement;

	const i = parseInt(element.id.substring(element.id.lastIndexOf('-') + 1, element.id.length));
	setMonthSlider(i);
}


/*==============================================================*/
/* Colour data
/*==============================================================*/
const MONTHS = {
	'short': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
	'medium': ['Jan.', 'Feb.', 'Mar.', 'April', 'May', 'June', 'July', 'Aug.', 'Sept.', 'Oct.', 'Nov.', 'Dec.'],
	'long': ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'December']
};
const KOPPEN = {
	'labels': ["Af – Tropical rainforest",
			  "Am – Tropical monsoon",
			  "Aw – Tropical savanna, dry winter",
			  "As – Tropical savanna, dry summer",
			  "BSh – Hot semi-arid",
			  "BSk – Cold semi-arid",
			  "BWh – Hot desert",
			  "BWk – Cold desert",
			  "Cfa – Humid subtropical",
			  "Cfb – Temperate oceanic",
			  "Cfc – Subpolar oceanic",
			  "Csa – Hot-summer Mediterranean",
			  "Csb – Warm-summer Mediterranean",
			  "Csc – Cold-summer Mediterranean",
			  "Cwa – Humid subtropical monsoon",
			  "Cwb – Temperate monsoon",
			  "Cwc – Subpolar oceanic monsoon",
			  "Dfa – Hot-summer humid continental",
			  "Dfb – Warm-summer humid continental",
			  "Dfc – Subarctic",
			  "Dfd – Extremely-cold subarctic",
			  "Dsa – Hot-summer Mediterranean humid continental",
			  "Dsb – Warm-summer Mediterranean humid continental",
			  "Dsc – Subarctic Mediterranean",
			  "Dsd – Extremely-cold Mediterranean arctic",
			  "Dwa – Hot-summer monsoon humid continental",
			  "Dwb – Warm-summer monsoon humid continental",
			  "Dwc – Subarctic monsoon",
			  "Dwd – Extremely-cold monsoon arctic",
			  "EF – Ice cap",
			  "ET – Tundra"],
	'colourMap': [[0, 0, 255], [0, 120, 255], [70, 170, 250], [35, 145, 255],
			   [245, 165, 0], [255, 220, 100], [255, 0, 0], [255, 150, 150],
			   [200, 255, 80], [100, 255, 80], [50, 200, 0],
			   [255, 255, 0], [200, 200, 0], [150, 150, 0],
			   [150, 255, 150], [100, 200, 100], [50, 150, 50],
			   [0, 255, 255], [55, 200, 255], [0, 125, 125], [0, 70, 95],
			   [255, 0, 255], [200, 0, 200], [150, 50, 150], [150, 100, 150],
			   [170, 175, 255], [90, 120, 220], [75, 80, 180], [50, 0, 135],
			   [102, 102, 102], [178, 178, 178]]
};
const TREWARTHA = {
	'labels': ["Ar – Tropical wet (rainforest)",
				"Am – Tropical monsoon",
				"Aw – Tropical wet & dry (savanna), dry winter",
				"As – Tropical wet & dry (savanna), dry summer",
				"BSh – Hot semi-arid (steppe)",
				"BSk – Cold semi-arid (steppe)",
				"BWh – Hot arid (desert)",
				"BWk – Cold arid (desert)",
				"Cfa – Hot humid subtropical",
				"Cfb – Warm humid subtropical",
				"Csa – Hot dry-summer subtropical",
				"Csb – Warm dry-summer subtropical",
				"Cwa – Hot dry-winter subtropical",
				"Cwb – Warm dry-winter subtropical",
				"DCfa – Hot humid temperate continental",
				"DCfb – Cold humid temperate continental",
				"DCsa – Hot dry-summer temperate continental",
				"DCsb – Cold dry-summer temperate continental",
				"DCwa – Hot dry-winter temperate continental",
				"DCwb – Cold dry-winter temperate continental",
				"DOfa – Hot humid temperate oceanic",
				"DOfb – Cold humid temperate oceanic",
				"DOsa – Hot dry-summer temperate oceanic",
				"DOsb – Cold dry-summer temperate oceanic",
				"DOwa – Hot dry-winter temperate oceanic",
				"DOwb – Cold dry-winter temperate oceanic",
				"EC – Boreal (subpolar) continental",
				"EO – Boreal (subpolar) oceanic",
				"Ft – Tundra",
				"Fi – Ice cap",
				"H – Highland"],
	'colourMap': [[0, 0, 255], [0, 120, 255], [70, 170, 250], [35, 145, 255],
				   [245, 165, 0], [255, 220, 100], [255, 0, 0], [255, 150, 150],
				   [200, 255, 80], [100, 255, 80],
				   [255, 255, 0], [200, 200, 0],
				   [150, 255, 150], [100, 200, 100],
				   [58, 111, 47], [90, 120, 220], [18, 86, 53], [69, 73, 180], [23, 90, 77], [100, 55, 181],
				   [85, 178, 67], [170, 175, 255], [32, 145, 93], [104, 108, 227], [87, 179, 176], [137, 72, 253],
				   [50, 0, 135], [0, 70, 147],
				   [178, 178, 178], [102, 102, 102],
				   [255, 204, 255]]
};
const COLOUR_MAPS = {
	'viridis': [[68.086, 1.243, 84.001], [69.821, 8.032, 91.508], [71.092, 15.847, 98.581], [71.891, 22.865, 105.166], [72.215, 29.498, 111.209], [71.984, 37.208, 117.685], [71.291, 43.503, 122.399], [70.174, 49.701, 126.481], [68.674, 55.799, 129.942], [66.446, 62.965, 133.321], [64.309, 68.795, 135.553], [61.994, 74.483, 137.322], [59.569, 80.026, 138.698], [57.101, 85.423, 139.754], [54.161, 91.719, 140.686], [51.781, 96.828, 141.251], [49.496, 101.827, 141.669], [47.317, 106.735, 141.972], [44.839, 112.529, 142.21], [42.872, 117.297, 142.311], [40.97, 122.028, 142.319], [39.108, 126.735, 142.22], [37.276, 131.43, 141.99], [35.131, 137.06, 141.501], [33.449, 141.754, 140.877], [31.975, 146.451, 140.017], [30.893, 151.148, 138.883], [30.468, 156.778, 137.111], [31.19, 161.454, 135.251], [33.167, 166.103, 133.01], [36.542, 170.712, 130.36], [41.314, 175.266, 127.278], [48.728, 180.633, 122.982], [56.132, 185.005, 118.888], [64.489, 189.264, 114.312], [73.675, 193.39, 109.249], [85.651, 198.14, 102.522], [96.334, 201.904, 96.374], [107.587, 205.472, 89.737], [119.354, 208.825, 82.619], [131.578, 211.945, 75.041], [146.769, 215.364, 65.386], [159.778, 217.934, 56.955], [173.015, 220.254, 48.323], [186.377, 222.339, 39.787], [202.409, 224.573, 30.601], [215.618, 226.267, 25.424], [228.562, 227.872, 24.565], [241.137, 229.453, 28.774], [253.278, 231.07, 36.704]],
	'plasma': [[12.848, 7.6, 134.634], [27.025, 6.199, 140.599], [37.895, 5.394, 145.493], [47.484, 4.795, 149.743], [56.405, 4.207, 153.531], [66.602, 3.394, 157.567], [74.837, 2.604, 160.52], [82.913, 1.763, 163.076], [90.872, 0.968, 165.192], [100.293, 0.284, 167.076], [108.041, 0.165, 168.034], [115.688, 0.703, 168.379], [123.219, 2.157, 168.069], [130.613, 4.802, 167.078], [139.27, 9.933, 164.988], [146.276, 15.307, 162.524], [153.068, 20.787, 159.462], [159.627, 26.345, 155.883], [167.173, 33.08, 151.041], [173.186, 38.721, 146.673], [178.951, 44.371, 142.11], [184.478, 50.02, 137.44], [189.779, 55.663, 132.734], [195.863, 62.428, 127.109], [200.719, 68.065, 122.478], [205.394, 73.71, 117.916], [209.899, 79.372, 113.426], [215.088, 86.206, 108.123], [219.236, 91.95, 103.764], [223.221, 97.753, 99.444], [227.037, 103.631, 95.148], [230.673, 109.598, 90.864], [234.782, 116.896, 85.722], [237.974, 123.109, 81.428], [240.935, 129.453, 77.12], [243.645, 135.939, 72.8], [246.533, 143.921, 67.605], [248.611, 150.747, 63.279], [250.366, 157.736, 58.978], [251.77, 164.891, 54.735], [252.798, 172.216, 50.606], [253.496, 181.228, 45.925], [253.596, 188.924, 42.415], [253.223, 196.789, 39.476], [252.345, 204.817, 37.365], [250.577, 214.662, 36.287], [248.483, 223.029, 36.736], [245.825, 231.527, 38.089], [242.69, 240.126, 38.996], [239.704, 248.665, 33.488]],
	'inferno': [[0.373, 0.119, 3.536], [1.957, 1.565, 11.943], [4.94, 3.859, 22.636], [9.605, 6.61, 33.719], [15.642, 9.33, 45.299], [23.712, 11.624, 59.761], [31.342, 12.122, 71.814], [39.742, 11.363, 82.961], [48.544, 10.024, 92.169], [59.042, 9.283, 100.062], [67.527, 10.11, 104.383], [75.78, 12.105, 107.225], [83.875, 14.746, 109.015], [91.872, 17.658, 110.032], [101.407, 21.231, 110.462], [109.336, 24.171, 110.265], [117.268, 27.053, 109.611], [125.211, 29.881, 108.516], [134.753, 33.237, 106.626], [142.704, 36.043, 104.57], [150.637, 38.904, 102.074], [158.53, 41.867, 99.139], [166.354, 44.987, 95.774], [175.607, 49.021, 91.189], [183.157, 52.692, 86.937], [190.517, 56.706, 82.328], [197.64, 61.115, 77.399], [205.806, 66.986, 71.119], [212.236, 72.398, 65.633], [218.278, 78.295, 59.959], [223.89, 84.675, 54.128], [229.039, 91.522, 48.159], [234.569, 100.314, 40.818], [238.615, 108.077, 34.537], [242.138, 116.193, 28.092], [245.13, 124.623, 21.494], [248.01, 135.098, 13.598], [249.815, 144.09, 8.009], [251.071, 153.286, 6.02], [251.771, 162.662, 8.904], [251.908, 172.193, 16.641], [251.319, 183.799, 28.618], [250.199, 193.579, 40.0], [248.533, 203.411, 52.615], [246.392, 213.229, 66.691], [243.524, 224.8, 86.056], [241.636, 233.937, 104.72], [241.724, 242.076, 125.314], [245.262, 248.861, 145.841], [252.032, 254.583, 164.456]],
	'magma': [[0.373, 0.119, 3.536], [1.935, 1.621, 11.468], [4.798, 4.087, 21.569], [9.058, 7.241, 31.928], [14.437, 10.751, 42.699], [21.279, 14.337, 56.293], [27.514, 16.405, 68.159], [34.439, 17.44, 80.325], [42.154, 17.317, 92.263], [52.258, 16.041, 104.936], [60.901, 15.177, 113.03], [69.358, 15.553, 118.743], [77.541, 17.298, 122.607], [85.504, 19.95, 125.211], [94.89, 23.668, 127.259], [102.65, 26.882, 128.363], [110.407, 30.053, 129.071], [118.195, 33.123, 129.451], [127.612, 36.648, 129.52], [135.534, 39.458, 129.258], [143.531, 42.169, 128.696], [151.6, 44.804, 127.816], [159.732, 47.396, 126.596], [169.553, 50.509, 124.653], [177.76, 53.168, 122.613], [185.95, 55.956, 120.176], [194.075, 58.96, 117.341], [203.645, 62.995, 113.436], [211.366, 66.868, 109.814], [218.73, 71.364, 105.951], [225.586, 76.635, 102.012], [231.765, 82.813, 98.254], [238.07, 91.485, 94.488], [242.304, 99.659, 92.429], [245.644, 108.474, 91.665], [248.212, 117.688, 92.301], [250.481, 128.992, 94.828], [251.85, 138.469, 98.229], [252.847, 147.927, 102.622], [253.549, 157.335, 107.852], [254.004, 166.683, 113.784], [254.282, 177.824, 121.681], [254.322, 187.054, 128.818], [254.209, 196.246, 136.397], [253.963, 205.409, 144.382], [253.527, 216.378, 154.452], [253.072, 225.504, 163.225], [252.595, 234.612, 172.297], [252.123, 243.714, 181.622], [251.699, 252.817, 191.124]],
	'cividis': [[0.0, 34.454, 77.712], [0.0, 38.197, 85.952], [0.0, 41.58, 94.76], [0.0, 44.901, 103.932], [0.0, 48.136, 112.089], [11.782, 51.995, 112.25], [24.389, 55.499, 111.331], [33.321, 59.022, 110.374], [40.732, 62.531, 109.53], [48.527, 66.719, 108.714], [54.425, 70.193, 108.242], [59.956, 73.656, 107.943], [65.211, 77.114, 107.818], [70.256, 80.571, 107.86], [76.097, 84.723, 108.113], [80.82, 88.19, 108.506], [85.444, 91.666, 109.039], [89.987, 95.154, 109.708], [95.34, 99.36, 110.723], [99.744, 102.883, 111.714], [104.098, 106.426, 112.855], [108.406, 109.99, 114.161], [112.662, 113.578, 115.683], [117.712, 117.913, 117.802], [121.937, 121.558, 119.555], [126.458, 125.224, 120.297], [131.128, 128.914, 120.523], [136.821, 133.382, 120.45], [141.625, 137.141, 120.142], [146.476, 140.934, 119.639], [151.374, 144.763, 118.932], [156.309, 148.63, 118.052], [162.285, 153.322, 116.744], [167.309, 157.279, 115.439], [172.375, 161.279, 113.918], [177.476, 165.325, 112.218], [183.657, 170.243, 109.855], [188.85, 174.394, 107.643], [194.079, 178.598, 105.223], [199.358, 182.854, 102.501], [204.68, 187.164, 99.489], [211.13, 192.411, 95.444], [216.552, 196.846, 91.731], [222.033, 201.341, 87.55], [227.572, 205.895, 82.867], [234.301, 211.438, 76.49], [239.992, 216.12, 70.333], [245.787, 220.853, 63.098], [251.76, 225.603, 54.401], [253.913, 231.883, 55.532]],
	'Greys': [[0.0, 0.0, 0.0], [5.804, 5.804, 5.804], [11.608, 11.608, 11.608], [17.412, 17.412, 17.412], [23.216, 23.216, 23.216], [30.18, 30.18, 30.18], [35.984, 35.984, 35.984], [42.824, 42.824, 42.824], [49.882, 49.882, 49.882], [58.353, 58.353, 58.353], [65.412, 65.412, 65.412], [72.471, 72.471, 72.471], [79.529, 79.529, 79.529], [85.365, 85.365, 85.365], [91.576, 91.576, 91.576], [96.753, 96.753, 96.753], [101.929, 101.929, 101.929], [107.106, 107.106, 107.106], [113.318, 113.318, 113.318], [118.706, 118.706, 118.706], [124.196, 124.196, 124.196], [129.686, 129.686, 129.686], [135.176, 135.176, 135.176], [141.765, 141.765, 141.765], [147.255, 147.255, 147.255], [153.059, 153.059, 153.059], [159.176, 159.176, 159.176], [166.518, 166.518, 166.518], [172.635, 172.635, 172.635], [178.753, 178.753, 178.753], [184.871, 184.871, 184.871], [190.427, 190.427, 190.427], [195.698, 195.698, 195.698], [200.09, 200.09, 200.09], [204.482, 204.482, 204.482], [208.875, 208.875, 208.875], [214.145, 214.145, 214.145], [218.263, 218.263, 218.263], [221.871, 221.871, 221.871], [225.478, 225.478, 225.478], [229.086, 229.086, 229.086], [233.416, 233.416, 233.416], [237.024, 237.024, 237.024], [240.412, 240.412, 240.412], [242.765, 242.765, 242.765], [245.588, 245.588, 245.588], [247.941, 247.941, 247.941], [250.294, 250.294, 250.294], [252.647, 252.647, 252.647], [255.0, 255.0, 255.0]],
	'Blues': [[247.0, 251.0, 255.0], [243.078, 248.49, 253.745], [239.157, 245.98, 252.49], [235.235, 243.471, 251.235], [231.314, 240.961, 249.98], [226.608, 237.949, 248.475], [222.686, 235.439, 247.22], [218.894, 232.929, 245.965], [215.129, 230.42, 244.71], [210.612, 227.408, 243.204], [206.847, 224.898, 241.949], [203.082, 222.388, 240.694], [199.318, 219.878, 239.439], [193.922, 217.267, 237.573], [186.392, 214.067, 234.937], [180.118, 211.4, 232.741], [173.843, 208.733, 230.545], [167.569, 206.067, 228.349], [160.039, 202.867, 225.714], [152.6, 199.035, 223.835], [144.6, 194.643, 222.11], [136.6, 190.251, 220.384], [128.6, 185.859, 218.659], [119.0, 180.588, 216.588], [111.0, 176.196, 214.863], [103.784, 171.804, 212.745], [97.353, 167.412, 210.235], [89.635, 162.141, 207.224], [83.204, 157.749, 204.714], [76.773, 153.357, 202.204], [70.341, 148.965, 199.694], [64.318, 144.318, 197.133], [58.106, 138.106, 193.933], [52.929, 132.929, 191.267], [47.753, 127.753, 188.6], [42.576, 122.576, 185.933], [36.365, 116.365, 182.733], [31.627, 111.243, 179.627], [27.706, 106.224, 175.706], [23.784, 101.204, 171.784], [19.863, 96.184, 167.863], [15.157, 90.161, 163.157], [11.235, 85.141, 159.235], [8.0, 80.094, 154.655], [8.0, 74.918, 146.969], [8.0, 68.706, 137.745], [8.0, 63.529, 130.059], [8.0, 58.353, 122.373], [8.0, 53.176, 114.686], [8.0, 48.0, 107.0]],
	'YlGnBu': [[255.0, 255.0, 217.0], [252.176, 253.902, 210.725], [249.353, 252.804, 204.451], [246.529, 251.706, 198.176], [243.706, 250.608, 191.902], [240.318, 249.29, 184.373], [237.494, 248.192, 178.098], [232.082, 246.059, 177.388], [226.122, 243.706, 177.859], [218.969, 240.882, 178.424], [213.008, 238.529, 178.894], [207.047, 236.176, 179.365], [201.086, 233.824, 179.835], [191.659, 230.145, 180.714], [178.106, 224.875, 182.031], [166.812, 220.482, 183.129], [155.518, 216.09, 184.227], [144.224, 211.698, 185.325], [130.671, 206.427, 186.643], [120.435, 202.565, 187.953], [110.71, 198.957, 189.365], [100.984, 195.349, 190.776], [91.259, 191.741, 192.188], [79.588, 187.412, 193.882], [69.863, 183.804, 195.294], [62.176, 179.098, 195.686], [56.529, 173.294, 195.059], [49.753, 166.329, 194.306], [44.106, 160.525, 193.678], [38.459, 154.722, 193.051], [32.812, 148.918, 192.424], [29.255, 142.4, 190.776], [30.196, 132.8, 186.259], [30.98, 124.8, 182.494], [31.765, 116.8, 178.729], [32.549, 108.8, 174.965], [33.49, 99.2, 170.447], [34.165, 91.694, 166.902], [34.635, 85.106, 163.765], [35.106, 78.518, 160.627], [35.576, 71.929, 157.49], [36.141, 64.024, 153.725], [36.612, 57.435, 150.588], [36.204, 51.369, 146.353], [31.655, 47.761, 136.941], [26.196, 43.431, 125.647], [21.647, 39.824, 116.235], [17.098, 36.216, 106.824], [12.549, 32.608, 97.412], [8.0, 29.0, 88.0]]
};
const COLOUR_MAPS_DIVERGING = {
	'RdBu': [[5.0, 48.0, 97.0], [10.49, 58.588, 111.706], [15.98, 69.176, 126.412], [21.471, 79.765, 141.118], [26.961, 90.353, 155.824], [33.667, 102.882, 172.451], [40.333, 111.706, 176.961], [47.0, 120.529, 181.471], [53.667, 129.353, 185.98], [61.667, 139.941, 191.392], [70.098, 148.961, 196.059], [85.588, 158.765, 201.353], [101.078, 168.569, 206.647], [116.569, 178.373, 211.941], [135.157, 190.137, 218.294], [149.706, 198.882, 223.059], [162.059, 205.157, 226.588], [174.412, 211.431, 230.118], [189.235, 218.961, 234.353], [201.588, 225.235, 237.882], [211.98, 230.412, 240.549], [219.431, 233.941, 241.922], [226.882, 237.471, 243.294], [235.824, 241.706, 244.941], [243.275, 245.235, 246.314], [247.588, 244.255, 242.294], [248.765, 238.765, 232.882], [250.176, 232.176, 221.588], [251.353, 226.686, 212.176], [252.529, 221.196, 202.765], [251.941, 212.647, 190.882], [250.176, 202.059, 177.353], [248.059, 189.353, 161.118], [246.294, 178.765, 147.588], [244.529, 168.176, 134.059], [239.882, 155.529, 122.725], [232.824, 139.294, 110.255], [226.941, 125.765, 99.863], [221.059, 112.235, 89.471], [215.176, 98.706, 79.078], [208.353, 84.706, 71.667], [199.882, 67.765, 63.667], [192.824, 53.647, 57.0], [185.765, 39.529, 50.333], [178.706, 25.412, 43.667], [161.824, 18.824, 40.412], [147.118, 14.118, 38.059], [132.412, 9.412, 35.706], [117.706, 4.706, 33.353], [103.0, 0.0, 31.0]],
	'RdYlBu': [[49.0, 54.0, 149.0], [52.922, 66.353, 155.078], [56.843, 78.706, 161.157], [60.765, 91.059, 167.235], [64.686, 103.412, 173.314], [69.922, 118.098, 180.569], [79.137, 129.078, 186.255], [88.353, 140.059, 191.941], [97.569, 151.039, 197.627], [108.627, 164.216, 204.451], [118.157, 174.725, 209.941], [128.941, 183.353, 214.647], [139.725, 191.98, 219.353], [150.51, 200.608, 224.059], [163.451, 210.961, 229.706], [174.118, 218.529, 233.882], [184.51, 223.627, 236.824], [194.902, 228.725, 239.765], [207.373, 234.843, 243.294], [217.765, 239.941, 246.235], [226.431, 243.941, 243.529], [232.51, 246.294, 232.353], [238.588, 248.647, 221.176], [245.882, 251.471, 207.765], [251.961, 253.824, 196.588], [254.902, 251.961, 186.392], [254.706, 245.882, 177.176], [254.471, 238.588, 166.118], [254.275, 232.51, 156.902], [254.078, 226.431, 147.686], [253.882, 218.118, 138.471], [253.686, 208.314, 129.255], [253.451, 196.549, 118.196], [253.255, 186.745, 108.98], [253.059, 176.941, 99.765], [251.765, 165.078, 92.882], [249.647, 149.784, 85.824], [247.882, 137.039, 79.941], [246.118, 124.294, 74.059], [244.353, 111.549, 68.176], [239.451, 99.431, 62.608], [232.627, 85.078, 56.02], [226.941, 73.118, 50.529], [221.255, 61.157, 45.039], [215.569, 49.196, 39.549], [204.216, 37.647, 38.784], [194.412, 28.235, 38.588], [184.608, 18.824, 38.392], [174.804, 9.412, 38.196], [165.0, 0.0, 38.0]],
	'Spectral': [[94.0, 79.0, 162.0], [85.373, 90.176, 167.294], [76.745, 101.353, 172.588], [68.118, 112.529, 177.882], [59.49, 123.706, 183.176], [51.02, 137.137, 188.529], [61.216, 148.51, 183.824], [71.412, 159.882, 179.118], [81.608, 171.255, 174.412], [93.843, 184.902, 168.765], [104.706, 195.059, 164.961], [118.235, 200.353, 164.765], [131.765, 205.647, 164.569], [145.294, 210.941, 164.373], [161.529, 217.294, 164.137], [174.471, 222.412, 163.294], [186.039, 227.118, 160.941], [197.608, 231.824, 158.588], [211.49, 237.471, 155.765], [223.059, 242.176, 153.412], [231.961, 245.784, 155.059], [236.863, 247.745, 162.706], [241.765, 249.706, 170.353], [247.647, 252.059, 179.529], [252.549, 254.02, 187.176], [254.902, 251.961, 185.902], [254.706, 245.882, 175.706], [254.471, 238.588, 163.471], [254.275, 232.51, 153.275], [254.078, 226.431, 143.078], [253.882, 218.118, 134.059], [253.686, 208.314, 125.824], [253.451, 196.549, 115.941], [253.255, 186.745, 107.706], [253.059, 176.941, 99.471], [251.765, 165.078, 92.882], [249.647, 149.784, 85.824], [247.882, 137.039, 79.941], [246.118, 124.294, 74.059], [244.353, 111.549, 68.176], [239.137, 101.627, 68.882], [231.843, 90.569, 71.706], [225.765, 81.353, 74.059], [219.686, 72.137, 76.412], [213.608, 62.922, 78.765], [201.137, 48.843, 76.196], [190.353, 36.882, 73.647], [179.569, 24.922, 71.098], [168.784, 12.961, 68.549], [158.0, 1.0, 66.0]],
	'coolwarm': [[58.6, 76.173, 192.189], [64.429, 84.873, 199.835], [70.336, 93.513, 207.201], [76.358, 102.063, 214.16], [82.548, 110.455, 220.504], [90.109, 120.378, 227.605], [96.543, 128.483, 232.991], [103.127, 136.334, 237.66], [109.779, 144.045, 241.967], [117.9, 152.957, 246.294], [124.758, 160.022, 249.108], [131.646, 166.897, 251.534], [138.577, 173.401, 253.228], [145.507, 179.548, 254.285], [153.806, 186.54, 254.889], [160.673, 191.892, 254.62], [167.454, 196.811, 253.703], [174.179, 201.461, 252.391], [182.032, 206.259, 249.744], [188.401, 209.756, 246.907], [194.658, 212.949, 243.693], [200.614, 215.426, 239.652], [206.379, 217.508, 235.192], [213.013, 219.431, 229.237], [218.121, 220.263, 223.53], [223.267, 219.362, 217.115], [228.45, 216.726, 209.992], [233.598, 212.664, 201.117], [237.32, 208.814, 193.582], [240.575, 204.58, 185.939], [242.954, 199.656, 178.155], [244.948, 194.438, 170.331], [246.575, 187.596, 160.874], [247.06, 181.264, 152.975], [247.169, 174.67, 145.089], [246.666, 167.655, 137.231], [245.207, 158.673, 127.896], [243.488, 150.864, 120.191], [241.193, 142.689, 112.586], [238.248, 134.109, 105.133], [234.959, 125.312, 97.769], [230.178, 114.224, 89.179], [225.584, 104.554, 82.239], [220.675, 94.638, 75.421], [215.144, 84.167, 68.867], [207.955, 70.834, 61.275], [201.593, 59.006, 55.142], [194.698, 45.56, 49.316], [187.445, 26.637, 43.731], [179.947, 3.967, 38.309]],
	'bwr': [[0.0, 0.0, 255.0], [10.0, 10.0, 255.0], [20.0, 20.0, 255.0], [30.0, 30.0, 255.0], [40.0, 40.0, 255.0], [52.0, 52.0, 255.0], [62.0, 62.0, 255.0], [72.0, 72.0, 255.0], [82.0, 82.0, 255.0], [94.0, 94.0, 255.0], [104.0, 104.0, 255.0], [114.0, 114.0, 255.0], [124.0, 124.0, 255.0], [134.0, 134.0, 255.0], [146.0, 146.0, 255.0], [156.0, 156.0, 255.0], [166.0, 166.0, 255.0], [176.0, 176.0, 255.0], [188.0, 188.0, 255.0], [198.0, 198.0, 255.0], [208.0, 208.0, 255.0], [218.0, 218.0, 255.0], [228.0, 228.0, 255.0], [240.0, 240.0, 255.0], [250.0, 250.0, 255.0], [255.0, 250.0, 250.0], [255.0, 240.0, 240.0], [255.0, 228.0, 228.0], [255.0, 218.0, 218.0], [255.0, 208.0, 208.0], [255.0, 198.0, 198.0], [255.0, 188.0, 188.0], [255.0, 176.0, 176.0], [255.0, 166.0, 166.0], [255.0, 156.0, 156.0], [255.0, 146.0, 146.0], [255.0, 134.0, 134.0], [255.0, 124.0, 124.0], [255.0, 114.0, 114.0], [255.0, 104.0, 104.0], [255.0, 94.0, 94.0], [255.0, 82.0, 82.0], [255.0, 72.0, 72.0], [255.0, 62.0, 62.0], [255.0, 52.0, 52.0], [255.0, 40.0, 40.0], [255.0, 30.0, 30.0], [255.0, 20.0, 20.0], [255.0, 10.0, 10.0], [255.0, 0.0, 0.0]],
	'seismic': [[0.0, 0.0, 76.5], [0.0, 0.0, 90.5], [0.0, 0.0, 104.5], [0.0, 0.0, 118.5], [0.0, 0.0, 132.5], [0.0, 0.0, 149.3], [0.0, 0.0, 163.3], [0.0, 0.0, 177.3], [0.0, 0.0, 191.3], [0.0, 0.0, 208.1], [0.0, 0.0, 222.1], [0.0, 0.0, 236.1], [0.0, 0.0, 250.1], [13.0, 13.0, 255.0], [37.0, 37.0, 255.0], [57.0, 57.0, 255.0], [77.0, 77.0, 255.0], [97.0, 97.0, 255.0], [121.0, 121.0, 255.0], [141.0, 141.0, 255.0], [161.0, 161.0, 255.0], [181.0, 181.0, 255.0], [201.0, 201.0, 255.0], [225.0, 225.0, 255.0], [245.0, 245.0, 255.0], [255.0, 245.0, 245.0], [255.0, 225.0, 225.0], [255.0, 201.0, 201.0], [255.0, 181.0, 181.0], [255.0, 161.0, 161.0], [255.0, 141.0, 141.0], [255.0, 121.0, 121.0], [255.0, 97.0, 97.0], [255.0, 77.0, 77.0], [255.0, 57.0, 57.0], [255.0, 37.0, 37.0], [255.0, 13.0, 13.0], [251.5, 0.0, 0.0], [241.5, 0.0, 0.0], [231.5, 0.0, 0.0], [221.5, 0.0, 0.0], [209.5, 0.0, 0.0], [199.5, 0.0, 0.0], [189.5, 0.0, 0.0], [179.5, 0.0, 0.0], [167.5, 0.0, 0.0], [157.5, 0.0, 0.0], [147.5, 0.0, 0.0], [137.5, 0.0, 0.0], [127.5, 0.0, 0.0]]
};
function getColour(map, x) {
	const i = x * (map.length - 1);
	const prev = Math.floor(i);
	const next = Math.ceil(i);
	const prevColour = map[prev];
	const nextColour = map[next];
	function interpolateColour(j) {
		const progress = i - prev;
		return Math.round(prevColour[j] * (1 - progress) + nextColour[j] * progress);
	}
	return [interpolateColour(0), interpolateColour(1), interpolateColour(2)];
}
function hexToRGB(hex) {
    const bigint = parseInt(hex.substring(1, hex.length), 16);
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    return [r, g, b];
}


/*==============================================================*/
/* Part 3 - Global climate
/*==============================================================*/
function drawResult() {
	if (CANVAS_DISPLAY == 'temperature')
		drawContinuous('temp');
	else if (CANVAS_DISPLAY == 'precipitation')
		drawContinuous('prec');
	else if (CANVAS_DISPLAY == 'koppen' || CANVAS_DISPLAY == 'trewartha')
		drawDiscrete(CANVAS_DISPLAY);
	else
		throw new Error(CANVAS_DISPLAY);
}
function drawContinuous(prefix) {
	const data = CLIMATE_NET[`${prefix}_${MONTHS.short[CURR_MONTH]}`];
	let [min, max] = CLIMATE_NET[`${prefix}_range`];
	if (prefix == 'prec')
		min = 0;
	const range = max - min;
	const name = getForm('colour-scheme');
	let colourMap = (name in COLOUR_MAPS) ? COLOUR_MAPS[name] : COLOUR_MAPS_DIVERGING[name];
	if (((name == 'Blues' || name == 'YlGnBu') && (prefix == 'temp')) || (name in COLOUR_MAPS_DIVERGING && prefix == 'prec'))
		colourMap = colourMap.toReversed();

	// Transformations
	const zeroCenteredColour = prefix == 'temp' && getForm('zero-centered-colour');
	const logarithmicScale = prefix == 'prec' && getForm('logarithmic-scale');
	const magnitude = Math.max(Math.abs(min), Math.abs(max));
	let f, getProgress;
	if (logarithmicScale)	f = x => Math.log(x + 1);
	else					f = x => x;
	if (zeroCenteredColour)	getProgress = (x, f) => (f(x) + f(magnitude)) / (f(magnitude) - f(-magnitude))
	else					getProgress = (x, f) => (f(x) - f(min)) / (f(max) - f(min));

	// Drawing
	const imageData = CONTEXT.getImageData(0, 0, W, H);
	const pixels = imageData.data;
	function colourPixels(offset, rgb) {
		pixels[offset] = rgb[0];
		pixels[offset + 1] = rgb[1];
		pixels[offset + 2] = rgb[2];
		pixels[offset + 3] = 255;
	}
	const waterRGB = hexToRGB(getForm('water-colour'));
	const longitudeOffset = getForm('brush-longitude-textnum');
	for (let i = 0; i < H; i++) {
		for (let j = 0; j < W; j++) { // J = actual matrix, j = drawing
			const offset = (i * W + j) * 4;
			const J = (j + longitudeOffset + W) % W;
			const value = data[i][J];
			const progress = bound(getProgress(value, f), 0, 1); // Numerical inaccuracies
			const water = CLIMATE_NET['water'][i][J];
			const colour = water ? waterRGB : getColour(colourMap, progress);
			colourPixels(offset, colour);
		}
	}
	CONTEXT.putImageData(imageData, 0, 0);
	drawColourBarContinuous(colourMap, min, max, f, getProgress);
}
function drawDiscrete(prefix) {
	const data = CLIMATE_NET[prefix];
	let system;
	if (prefix == 'koppen')
		system = KOPPEN;
	else if (prefix == 'trewartha')
		system = TREWARTHA;
	else
		throw new Error(prefix);

	// Drawing
	const imageData = CONTEXT.getImageData(0, 0, W, H);
	const pixels = imageData.data;
	function colourPixels(offset, rgb) {
		pixels[offset] = rgb[0];
		pixels[offset + 1] = rgb[1];
		pixels[offset + 2] = rgb[2];
		pixels[offset + 3] = 255;
	}
	const waterRGB = [0, 0, 0];
	const longitudeOffset = getForm('brush-longitude-textnum');
	for (let i = 0; i < H; i++) {
		for (let j = 0; j < W; j++) { // J = actual matrix, j = drawing
			const offset = (i * W + j) * 4;
			const J = (j + longitudeOffset + W) % W;
			const value = data[i][J];
			const water = CLIMATE_NET['water'][i][J];
			const colour = water ? waterRGB : system.colourMap[value];
			colourPixels(offset, colour);
		}
	}
	CONTEXT.putImageData(imageData, 0, 0);
	drawColourBarDiscrete(system);
}

/*==============================================================*/
/* Display helper functions
/*==============================================================*/
function maxTextWidth(context, arr, size, decimal, font, format) {
	context.font = `${format} ${size}px "${font}"`;
	f = x => context.measureText(x.toFixed(decimal)).width;
	return arr.map(f).reduce((max, curr) => Math.max(max, curr), -Infinity);
}

function getTicks(bounds, amount) {
	const [min, max] = bounds;
	const range = max - min;
	const minTicks = 3;
	const maxTicks = 11;
	const minIncrement = range / (maxTicks - 1);
	const maxIncrement = range / (minTicks - 1);
	let increment = minIncrement + (1 - amount) * (maxIncrement - minIncrement);

	const exponent = Math.floor(Math.log10(increment));
	const candidates = [0.5, 1, 2, 5, 10];
	const increments = candidates.map(x => x * (10 ** exponent));
	let bestDistance = Infinity;
	let bestTicks, bestIncrements, bestRange;
	for (let i = 0; i < candidates.length; i++) {
		let candidate = increments[i];

		// Round to nearest increment
		let minDisplay = roundToFloat(min, 1 / candidate);
		if (minDisplay > min)
			minDisplay -= candidate;
		let maxDisplay = roundToFloat(max, 1 / candidate);
		if (maxDisplay < max)
			maxDisplay += candidate;

		// Extend axis to 0
		let ticks = (maxDisplay - minDisplay) / candidate + 1;
		if (minDisplay > 0 && (minDisplay / candidate) / ticks < 0.25) {
			minDisplay = 0;
			ticks = maxDisplay / candidate + 1;
		}
		if (maxDisplay < 0 && (maxDisplay / candidate) / ticks > -0.25) {
			maxDisplay = 0;
			ticks = -minDisplay / candidate + 1;
		}

		const distance = Math.abs(candidate - increment);
		const eligible = (minTicks <= ticks) && (ticks <= maxTicks)
		if (distance < bestDistance && eligible) {
			bestDistance = distance;
			bestIncrement = candidate;
			bestRange = [minDisplay, maxDisplay];
		}
	}
	if (bestDistance == Infinity)
		throw new Error("Welp...");
	increment = bestIncrement;
	const [minDisplay, maxDisplay] = bestRange;
	const rangeDisplay = maxDisplay - minDisplay;
	const n = rangeDisplay / increment + 1;
	const ticks = [];
	for (let i = 0; i < n; i += 1) {
		ticks.push(minDisplay + rangeDisplay * i / (n - 1));
	}
	return [ticks, increment, minDisplay, maxDisplay];
}


/*==============================================================*/
/* Display functions
/*==============================================================*/
function drawText(context, text, x, y, size, align, font, colour, rotation, format) {
	context.save();
	context.translate(x, y);
	context.rotate(rotation * Math.PI / 180);
	context.fillStyle = colour;
	context.font = `${format} ${size}px "${font}"`;
	context.textAlign = align;
	context.textBaseline = 'middle';
	context.fillText(text, 0, 0);
	context.restore();
}
function drawCircle(context, x, y, radius, colour) {
	context.beginPath();
	context.arc(x, y, radius, 0, 2 * Math.PI);
	context.fillStyle = colour;
	context.fill();
	context.closePath();
}
function drawLine(context, startX, startY, endX, endY, thickness, colour) {
	context.lineWidth = thickness;
	context.strokeStyle = colour;
	context.beginPath();
	context.moveTo(startX, startY);
	context.lineTo(endX, endY);
	context.stroke();
	context.closePath();
}
function fillCanvas(canvas, context, colour) {
	context.fillStyle = colour;
	context.fillRect(0, 0, canvas.width, canvas.height);
}


/*==============================================================*/
/* Part 3 - Local climate
/*==============================================================*/
let LOCAL_PIXEL = null;
let LOCAL_TEMPERATURE_POINTS = [];
let LOCAL_PRECIPITATION_BINS = [];


function resetLocalClimate() {
	toggleLocalCanvas(false);
	LOCAL_PIXEL = null;
	cssSetId('option-local-climate', 'display', 'none');
	cssGetId('local-pixel').innerText = '';
	cssGetId('local-similar-cities').innerText = '';

	// Download output, local plot option
	const selector = cssFindFirst('#download-png-item .download-selector-active');
	if (selector.innerText == 'Local Climate')
		selector.innerText = 'Köppen-Geiger';
}
function initCanvas(id, text) {
	const canvas = cssGetId(id);
	const context = canvas.getContext("2d", {willReadFrequently: false});

	const bgColour = 'rgb(100, 120, 140)';
	const colour = 'rgb(180, 190, 200)';
	const font = 'GFS Neohellenic';
	const textSize = canvas.height / 7;
	const x = canvas.width / 2;
	const y = canvas.height / 2;

	fillCanvas(canvas, context, bgColour);
	drawText(context, text, x, y, textSize, 'center', font, colour, 0, '');
}
function hoverLocalCanvas(event) {
	trackMouse(event, cssGetId('local-canvas'));
	const threshold = 20;
	const result = [null, null, null];

	// Temperature point
	let closest1 = [Infinity, null];
	let closest2 = [Infinity, null];
	for (let i = 2; i < LOCAL_TEMPERATURE_POINTS.length; i++) {
		const [x, y] = LOCAL_TEMPERATURE_POINTS[i];
		const distance = euclideanDistance(PIXEL_X, PIXEL_Y, x, y);
		if (distance < threshold * 2 && result[0] == null)
			result[0] = [distance, i - 2];
		if (distance < closest1[0]) {
			closest2 = closest1;
			closest1 = [distance, i];
		} else if (distance < closest2[0]) {
			closest2 = [distance, i];
		}
	}
	// Temperature line
	const [x1, y1] = LOCAL_TEMPERATURE_POINTS[closest1[1]];
	const [x2, y2] = LOCAL_TEMPERATURE_POINTS[closest2[1]];
	const distance = distanceToLine(PIXEL_X, PIXEL_Y, x1, y1, x2, y2);
	if (distance < threshold)
		result[1] = distance;

	// Precipitation bar
	for (let i = 0; i < LOCAL_PRECIPITATION_BINS.length; i++) {
		const [left, right, top, bottom] = LOCAL_PRECIPITATION_BINS[i];
		if ((left <= PIXEL_X) && (PIXEL_X <= right) && (top <= PIXEL_Y) && (PIXEL_Y <= bottom)) {
			result[2] = i;
			break;
		}
	}

	// Use stricter threshold if temperature and precipitation overlap
	const strictThreshold = 12;
	const [point, line, bar] = result;
	if (bar != null && (point != null || line != null)) {
		if ((point != null) && (point[0] < strictThreshold))
			return plotLocalClimate(point[1], -100);
		else if ((line != null) && (line < strictThreshold))
			return plotLocalClimate(-1, -100);
		else
			return plotLocalClimate(-100, bar);
	}
	if (point != null) {
		return plotLocalClimate(point[1], -100);
	} if (line != null)
		return plotLocalClimate(-1, -100);
	if (bar != null)
		return plotLocalClimate(-100, bar);
	return plotLocalClimate(-100, -100);
}
function plotLocalClimate(hoverTemp, hoverPrec) {
	if (hoverTemp < -1 && hoverPrec < -1) {
		LOCAL_TEMPERATURE_POINTS.length = 0;
		LOCAL_PRECIPITATION_BINS.length = 0;
		cssSetId('option-local-climate', 'display', 'block');
	}

	const [X, Y] = LOCAL_PIXEL;
	const monthInds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
	const tempAll = monthInds.map(i => CLIMATE_NET[`temp_${MONTHS['short'][i]}`][Y][X]);
	const precAll = monthInds.map(i => CLIMATE_NET[`prec_${MONTHS['short'][i]}`][Y][X]);
	const tempBounds = arrayBounds1D(tempAll);
	const precBounds = arrayBounds1D(precAll);
	plotLocalClimateText(X, Y, tempAll, precAll);

	let canvas = cssGetId('local-canvas');
	if (canvas.style['display'] != 'block')
		canvas = cssGetId('local-mini-canvas');
	const context = canvas.getContext("2d", {willReadFrequently: true, imageSmoothingEnabled: true});
	const bgColour = 'white';
	const colour = 'black';
	const font = 'Carlito';
	const textSize = 50;
	const padding = 50; // Edge to text
	const textPadding = 10; // Text to axis
	const axesColour = 'black';
	const axesThickness = 6;
	const tickColour = 'black';
	const tickThickness = 6;
	const tickLength = 10;
	const gridColour = 'rgba(0, 0, 0, 0.1)';
	const gridThickness = 4;
	const pointRadius = (hoverTemp >= -1) ? 15 : 12;
	const pointColour = (hoverTemp >= -1) ? 'rgb(255, 111, 100)' : 'rgb(254, 57, 57)';
	const lineColour = (hoverTemp >= -1) ? 'rgb(255, 111, 100)' : 'rgb(254, 57, 57)';
	const lineThickness = (hoverTemp >= -1) ? 15 : 12;
	const barColour = (hoverPrec >= 0) ? 'rgb(80, 230, 255)' : 'rgb(50, 220, 255)';
	const barWidth = 0.5;

	fillCanvas(canvas, context, bgColour);

	// Axes
	const [tempTicks, tempIncrement, tempMin, tempMax] = getTicks(tempBounds, 0.5);
	const [precTicks, precIncrement, precMin, precMax] = getTicks(precBounds, 0.5);
	const tempRange = tempMax - tempMin;
	const precRange = precMax - precMin;
	const tempDecimals = Math.max(0, -Math.floor(Math.log10(tempRange)));
	const precDecimals = Math.max(0, -Math.floor(Math.log10(precRange)));
	const tempPadding = maxTextWidth(context, tempTicks, textSize, tempDecimals, font, '');
	const precPadding = maxTextWidth(context, precTicks, textSize, precDecimals, font, '');

	// Outer canvas
	const yAxis1Left = padding + (textSize + textPadding) + (tempPadding + textPadding) + tickLength;
	const yAxis1Right = yAxis1Left + axesThickness;
	const yAxis2Right = canvas.width - padding - (textSize + textPadding) - (precPadding + textPadding) - tickLength;
	const yAxis2Left = yAxis2Right - axesThickness;
	const xAxisBottom = canvas.height - padding - 2 * (textSize + textPadding) - tickLength;
	const xAxisTop = xAxisBottom - axesThickness;
	const axesTop = padding + textSize;

	// Draw axes
	context.fillStyle = axesColour;
	context.fillRect(yAxis1Left, axesTop, axesThickness, xAxisBottom - axesTop);
	context.fillRect(yAxis2Left, axesTop, axesThickness, xAxisBottom - axesTop);
	context.fillRect(yAxis1Left, xAxisTop, yAxis2Right - yAxis1Left, axesThickness);

	// Inner canvas
	const leftExtent = yAxis1Right;
	const rightExtent = yAxis2Left;
	const topExtent = axesTop + axesThickness;
	const bottomExtent = xAxisTop;
	const height = bottomExtent - topExtent;
	const width = rightExtent - leftExtent;
	const tempToY = temp => topExtent + (1 - (temp - tempMin) / tempRange) * height;
	const precToY = prec => topExtent + (1 - (prec - precMin) / precRange) * height;
	const monthToX = month => leftExtent + ((month + 0.5) / 12) * width;

	// Draw temperature axis
	let textX = yAxis1Left - tickLength - textPadding;
	for (let i = 0; i < tempTicks.length; i++) {
		const temp = tempTicks[i];
		const y = tempToY(temp);
		context.fillStyle = tickColour;
		context.fillRect(yAxis1Left - tickLength, y - tickThickness / 2, tickLength, tickThickness);
		drawText(context, temp.toFixed(tempDecimals), textX, y, textSize, 'right', font, colour, 0, '');
	}
	// Draw precipitation axis
	textX = yAxis2Right + tickLength + textPadding;
	for (let i = 0; i < precTicks.length; i++) {
		const prec = precTicks[i];
		const y = precToY(prec);
		context.fillStyle = tickColour;
		context.fillRect(yAxis2Right, y - tickThickness / 2, tickLength, tickThickness);
		drawText(context, prec.toFixed(precDecimals), textX, y, textSize, 'left', font, colour, 0, '');
	}
	// Draw month axis
	const textY = xAxisBottom + tickLength + textPadding + textSize / 2;
	for (let i = 0; i < MONTHS['medium'].length; i++) {
		const month = MONTHS['medium'][i];
		const x = monthToX(i);
		context.fillStyle = tickColour;
		context.fillRect(x - tickThickness / 2, xAxisBottom, tickThickness, tickLength);
		context.fillStyle = gridColour;
		context.fillRect(x - tickThickness / 2, topExtent, tickThickness, height);
		drawText(context, month, x, textY, textSize, 'center', font, colour, 0, '');
	}
	// Draw axis titles
	const yAxisTitleY = (topExtent + bottomExtent) / 2;
	const tempAxisTitleX = padding + textSize / 2;
	const precAxisTitleX = canvas.width - tempAxisTitleX;
	const xAxisTitleX = (leftExtent + rightExtent) / 2;
	const xAxisTitleY = canvas.height - tempAxisTitleX;
	drawText(context, 'Temperature', tempAxisTitleX, yAxisTitleY, textSize, 'center', font, colour, 270, '');
	drawText(context, 'Precipitation', precAxisTitleX, yAxisTitleY, textSize, 'center', font, colour, 90, '');
	drawText(context, 'Month', xAxisTitleX, xAxisTitleY, textSize, 'center', font, colour, 0, '');

	// Draw precipitation
	context.fillStyle = barColour;
	const barWidthPx = (monthToX(1) - monthToX(0)) * barWidth;
	for (let i = 0; i < precAll.length; i++) {
		const prec = precAll[i];
		const x = monthToX(i);
		const y = precToY(prec);
		context.fillRect(x - barWidthPx / 2, y, barWidthPx, bottomExtent - y);
		if (hoverTemp < -1 && hoverPrec < -1) // left, right, top, bottom
			LOCAL_PRECIPITATION_BINS.push([x - barWidthPx / 2, x + barWidthPx / 2, y, bottomExtent]);
	}
	// Draw temperature
	const borderValue = (tempAll[0] + tempAll[tempAll.length - 1]) / 2;
	const borderY = tempToY(borderValue);
	if (hoverTemp < -1 && hoverPrec < -1) {
		LOCAL_TEMPERATURE_POINTS.push([monthToX(-0.5), borderY]);
		LOCAL_TEMPERATURE_POINTS.push([monthToX(11.5), borderY]);
	}
	for (let i = 0; i < tempAll.length; i++) {
		const temp = tempAll[i];
		const x = monthToX(i);
		const y = tempToY(temp);
		if (i == 0)
			drawLine(context, monthToX(-0.5), borderY, x, y, lineThickness, lineColour);
		if (i < tempAll.length - 1)
			drawLine(context, x, y, monthToX(i + 1), tempToY(tempAll[i + 1]), lineThickness, lineColour);
		if (i == tempAll.length - 1)
			drawLine(context, x, y, monthToX(11.5), borderY, lineThickness, lineColour);
		drawCircle(context, x, y, pointRadius, pointColour);
		if (hoverTemp < -1 && hoverPrec < -1)
			LOCAL_TEMPERATURE_POINTS.push([x, y]);
	}

	function drawCaption(x, y, title, value) {
		const captionDistance = 60;
		const captionColour = 'rgba(22, 22, 22, 0.8)';
		const captionPaddingVertical = 25;
		const captionPaddingHorizontal = 25;
		const triangleWidth = 20;
		const triangleHeight = 50;
		const titleSize = 75;
		const valueSize = 50;
		const titleColour = 'white';
		const valueColour = 'white';

		context.font = `bold ${titleSize}px "${font}"`;
		const titleWidth = context.measureText(title).width;
		context.font = `${valueSize}px "${font}"`;
		const valueWidth = context.measureText(value).width;
		const captionWidth = 2 * captionPaddingHorizontal + Math.max(titleWidth, valueWidth);
		const captionHeight = 2 * captionPaddingVertical + titleSize + valueSize; + textPadding;
		const captionTop = y - captionHeight / 2;
		const closestCaptionDistance = 50;
		let captionLeft = x + captionDistance;
		let captionRight = captionLeft + captionWidth;
		let trianglePoint = captionLeft - triangleWidth;
		let triangleEnd = captionLeft;
		if (captionRight > canvas.width - closestCaptionDistance) {
			captionRight = x - captionDistance;
			captionLeft = captionRight - captionWidth;
			trianglePoint = captionRight + triangleWidth;
			triangleEnd = captionRight;
		}

		// Triangle and caption
		context.fillStyle = captionColour;
		context.beginPath();
		context.moveTo(trianglePoint, y);
		context.lineTo(triangleEnd, y - triangleHeight / 2);
		context.lineTo(triangleEnd, y + triangleHeight / 2);
		context.fill();
		context.closePath();
		context.beginPath();
		context.roundRect(captionLeft, captionTop, captionWidth, captionHeight, 10);
		context.fill();
		context.closePath();

		// Draw text
		const textX = captionLeft + captionPaddingHorizontal;
		const titleY = captionTop + captionPaddingVertical + titleSize / 2;
		const valueY = titleY + titleSize / 2 + valueSize / 2;
		drawText(context, title, textX, titleY, titleSize, 'left', font, titleColour, 0, 'bold');
		drawText(context, value, textX, valueY, valueSize, 'left', font, valueColour, 0, '');
	}
	if (hoverTemp >= 0) {
		const temp = tempAll[hoverTemp];
		const [x, y] = LOCAL_TEMPERATURE_POINTS[hoverTemp + 2];
		drawCaption(x, y, `${temp.toFixed(2)}˚C`, `Temperature`);
	} else if (hoverPrec >= 0) {
		const prec = precAll[hoverPrec];
		const x = (LOCAL_PRECIPITATION_BINS[hoverPrec][0] + LOCAL_PRECIPITATION_BINS[hoverPrec][1]) / 2;
		const y = (LOCAL_PRECIPITATION_BINS[hoverPrec][2] + LOCAL_PRECIPITATION_BINS[hoverPrec][3]) / 2;
		drawCaption(x, y, `${prec.toFixed(2)} mm`, `Precipitation`);
	}
}
function plotLocalClimateText(x, y, tempAll, precAll) {
	const tempSorted = tempAll.toSorted((a, b) => a - b);
	const precSorted = precAll.toSorted((a, b) => a - b);
	function getLoss(arr, x, offset, lossFunction) {
		let sum = 0;
		let totalArr = 0;
		let totalPoint = 0;
		for (let i = 0; i < 12; i++) {
			const v1 = parseFloat(arr[i]);
			const v2 = parseFloat(x[i + offset]);
			totalArr += v1;
			totalPoint += v2;
			sum += lossFunction(v1, v2);
		}
		return [sum, (totalArr - totalPoint) / 12];
	}
	const iTemp0 = CSV_HEADER.indexOf('temp0');
	const iPrec0 = CSV_HEADER.indexOf('prec0');
	const losses = CSV.map(x => {
		const [tempLoss, tempAvgLoss] = getLoss(tempSorted, x, iTemp0, (x, y) => Math.abs(x - y));
		const [precLoss, precAvgLoss] = getLoss(precSorted, x, iPrec0, (x, y) => (x ** 0.5 - y ** 0.5) ** 2);
		return [tempLoss, precLoss, tempAvgLoss, precAvgLoss, tempLoss + precLoss];
	});
	const ind = argsort(losses, 4);

	const iName = CSV_HEADER.indexOf('city');
	const iLon = CSV_HEADER.indexOf('longitude');
	const iLat = CSV_HEADER.indexOf('latitude');
	const iCountry = CSV_HEADER.indexOf('country');
	const getLongitude = (x, d) => `${Math.abs(180 - x).toFixed(d)}˚ ${x <= 180 ? 'W' : 'E'}`;
	const getLatitude = (y, d) => `${Math.abs(90 - y).toFixed(d)}˚ ${y <= 90 ? 'N' : 'S'}`;

	const values = []
	const maxDisplay = 10;
	const maxTempAvgDist = 3;
	const maxPrecAvgDist = 10;
	for (let i = 0; i < ind.length && values.length < maxDisplay; i++) {
		const [tempLoss, precLoss, tempAvgLoss, precAvgLoss, loss] = losses[ind[i]];
		if (Math.abs(tempAvgLoss) <= maxTempAvgDist && Math.abs(precAvgLoss) <= maxPrecAvgDist)
			values.push(ind[i]);
	}

	let text = '<p><i>Climate classification:</i></p>';
	text += `<p>${KOPPEN.labels[CLIMATE_NET['koppen'][y][x]]}<span> // Köppen</span></p>`;
	text += `<p>${TREWARTHA.labels[CLIMATE_NET['trewartha'][y][x]]}<span> // Trewartha</span></p><br>`;
	text += '<p><i>Similar-climate cities:</i></p>';
	if (values.length == 0)
		text += 'None :(';
	for (let i = 0; i < values.length; i++) {
		const row = CSV[values[i]];
		const longitude = getLongitude(row[iLon], 2);
		const latitude = getLatitude(row[iLat], 2);
		const tempDiff = losses[values[i]][2];
		const precDiff = losses[values[i]][3];
		const temp = tempDiff > 0 ? `+${tempDiff.toFixed(2)}` : `${tempDiff.toFixed(2)}`;
		const prec = precDiff > 0 ? `+${precDiff.toFixed(2)}` : `${precDiff.toFixed(2)}`;
		text += `<p><b>${row[iName]}</b>, ${row[iCountry]} <span>(${longitude}, ${latitude})</span></p><div>ΔTₘₑₐₙ = ${temp}</div><div>ΔPₘₑₐₙ = ${prec}</div>`;
	}
	cssGetId('local-similar-cities').innerHTML = text;
	cssGetId('local-pixel').innerText = `Pixel at (${getLongitude(x, 0)}, ${getLatitude(y, 0)})`;
}



/*==============================================================*/
/* Part 3 - Colour bar
/*==============================================================*/
addEventListener("resize", (event) => {
	if (CURR_PART != 3)
		return;

	// Dropdown heights
	for (let element of cssGetClass('download-selector')) {
		const active = cssFindFirst(`#${element.id} .download-selector-active`);
		const options = cssFindFirst(`#${element.id} .download-selector-options`);
		const maxHeight = window.innerHeight - active.getBoundingClientRect().bottom;
		options.style.setProperty('max-height', `${maxHeight}px`);
	}
});
function drawColourBarText(textContainer, text, y) {
	const p = document.createElement("p");
	const node = document.createTextNode(text);
	p.appendChild(node);
	p.style.setProperty('top', `${100 * y / textContainer.getBoundingClientRect().height}%`);
	textContainer.appendChild(p);
}
function drawColourBarContinuous(colourMap, min, max, f, getProgress) {
	const canvas = cssGetId('colour-bar-canvas');
	const context = canvas.getContext("2d", {willReadFrequently: false});
	const textContainer = cssGetId('colour-bar-text');
	textContainer.textContent = '';

	const toRGB = (x) => `rgb(${x[0]}, ${x[1]}, ${x[2]})`;
	const start = getProgress(min, x => x);
	const end = getProgress(max, x => x);

	// Draw gradient
	let n = colourMap.length;
	const gradient = context.createLinearGradient(0, canvas.height, 0, 0);
	for (let i = 0; i < n; i ++) {
		const x = i / (n - 1);
		const progress = getProgress(min + x * (max - min), x => x);
		gradient.addColorStop(x, toRGB(getColour(colourMap, progress)));
	}
	context.fillStyle = gradient;
	context.fillRect(0, 0, canvas.width, canvas.height);

	// Draw text
	context.fillStyle = 'rgb(22, 22, 22)';
	const canvasPhysicalHeight = canvas.getBoundingClientRect().height;
	const [ticks, increment, minDisplay, maxDisplay] = getTicks([min, max], 1);
	for (let i = 0; i < ticks.length; i++) {
		const value = ticks[i];
		if (value < min || value > max)
			continue;
		const progress = 1 - (f(value) - f(min)) / (f(max) - f(min));
		const y = progress * canvasPhysicalHeight;
		drawColourBarText(textContainer, value, y - 6);

		// Ticks
		const yTick = progress * canvas.height;
		context.fillRect(0, yTick, 5, 1);
		context.fillRect(canvas.width - 5, yTick, 5, 1);
	}
}
function drawColourBarDiscrete(system) {
	const canvas = cssGetId('colour-bar-canvas');
	const context = canvas.getContext("2d", {willReadFrequently: false});
	const textContainer = cssGetId('colour-bar-text');
	textContainer.textContent = '';

	const toRGB = (x) => `rgb(${x[0]}, ${x[1]}, ${x[2]})`;
	const canvasPhysicalHeight = canvas.getBoundingClientRect().height;
	const n = system.colourMap.length;
	for (let i = 0; i < n; i++) {
		// Fill rectangle
		const yStart = Math.round(i * canvas.height / n);
		const yEnd = Math.round((i + 1) * canvas.height / n);
		context.fillStyle = toRGB(system.colourMap[i]);
		context.fillRect(0, yStart, canvas.width, yEnd - yStart);

		// Draw text
		const yText = canvasPhysicalHeight * (i + 0.5) / n - 6;
		const text = system.labels[i].substring(0, system.labels[i].search(' –'));
		drawColourBarText(textContainer, text, yText);
	}
}


/*==============================================================*/
/* Part 3 - Statistics Histograms
/*==============================================================*/
function plotStatisticsHistograms() {
	const latitudeHistogram = getLatitudeHistogram(DATA_WATER);
	if (isNaN(latitudeHistogram[0])) {
		initCanvas('elevation-canvas', 'There is no land to plot');
		initCanvas('latitude-canvas', 'There is no land to plot');
		return;
	}
	const latitudeTicks = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90];
	plotHistogram('Absolute Latitude (°)', 'Land Area (%)', 'latitude-canvas', latitudeTicks, 0, 1, latitudeHistogram);

	const binIncrement = 100;
	const [elevationHistogram, binMin, binMax] = getElevationHistogram(DATA_ELEVATION, DATA_WATER, DATA_FINAL_BOUNDS, binIncrement);
	const elevationTicks = getTicks(DATA_FINAL_BOUNDS, 0.5)[0];
	plotHistogram('Elevation (m)', 'Land Area (%)', 'elevation-canvas', elevationTicks, binMin, binIncrement, elevationHistogram);
}
function plotHistogram(xTitle, yTitle, canvasName, xTicks, binMin, binIncrement, binValues) {
	const canvas = cssGetId(canvasName);
	const context = canvas.getContext("2d", {willReadFrequently: false, imageSmoothingEnabled: true});
	const bgColour = 'white';
	const colour = 'black';
	const font = 'Carlito';
	const textSize = 75;
	const padding = 75; // Edge to text
	const textPadding = 10; // Text to axis
	const axesColour = 'black';
	const axesThickness = 10;
	const tickColour = 'black';
	const tickThickness = 10;
	const tickLength = 15;
	const gridColour = 'rgba(0, 0, 0, 0.1)';
	const gridThickness = 6;
	const barColour = 'rgb(15, 82, 186)';

	fillCanvas(canvas, context, bgColour);

	// Axes
	const [xMin, xMax] = [xTicks[0], xTicks[xTicks.length - 1]];
	const xRange = xMax - xMin;
	const yBounds = [0, binValues.reduce((max, curr) => Math.max(max, curr), -Infinity)];
	const [yTicks, yIncrement, yMin, yMax] = getTicks(yBounds, 0.5);
	const yRange = yMax - yMin;
	const yTicksDisplay = yTicks.map(x => x * 100);
	const yDecimals = Math.max(0, -Math.floor(Math.log10(yIncrement * 100)));
	const yPadding = maxTextWidth(context, yTicksDisplay, textSize, yDecimals, font, '');

	// Outer canvas
	const yAxisLeft = padding + (textSize + textPadding) + (yPadding + textPadding) + tickLength;
	const yAxisRight = yAxisLeft + axesThickness;
	const xAxisBottom = canvas.height - padding - 2 * (textSize + textPadding) - tickLength;
	const xAxisTop = xAxisBottom - axesThickness;
	const axesTop = padding + textSize;
	const axesRight = canvas.width - padding - textSize;

	// Draw axes
	context.fillStyle = axesColour;
	context.fillRect(yAxisLeft, axesTop, axesThickness, xAxisBottom - axesTop);
	context.fillRect(yAxisLeft, xAxisTop, axesRight - yAxisLeft, axesThickness);

	// Inner canvas
	const leftExtent = yAxisRight;
	const rightExtent = axesRight - axesThickness;
	const topExtent = axesTop + axesThickness;
	const bottomExtent = xAxisTop;
	const height = bottomExtent - topExtent;
	const width = rightExtent - leftExtent;
	const valueToY = value => topExtent + (1 - (value - yMin) / yRange) * height;
	const valueToX = value => leftExtent + ((value - xMin) / xRange) * width;

	// Draw y-axis
	const textX = yAxisLeft - (tickLength + textPadding);
	for (let i = 0; i < yTicks.length; i++) {
		const value = yTicks[i];
		const y = valueToY(value);
		context.fillStyle = tickColour;
		context.fillRect(yAxisLeft - tickLength, y - tickThickness / 2, tickLength, tickThickness);
		if (value != yTicks[0]) {
			context.fillStyle = gridColour;
			context.fillRect(leftExtent, y - gridThickness / 2, width, gridThickness);
		}
		drawText(context, yTicksDisplay[i].toFixed(yDecimals), textX, y, textSize, 'right', font, colour, 0, '');
	}
	// Draw x-axis
	const textY = xAxisBottom + tickLength + textPadding + textSize / 2;
	for (let i = 0; i < xTicks.length; i++) {
		const value = xTicks[i];
		const x = valueToX(value);
		context.fillStyle = tickColour;
		context.fillRect(x - tickThickness / 2, xAxisBottom, tickThickness, tickLength);

		if (value != xTicks[0]) {
			context.fillStyle = gridColour;
			context.fillRect(x - gridThickness / 2, topExtent, gridThickness, height);
		}
		drawText(context, value, x, textY, textSize, 'center', font, colour, 0, '');
	}

	// Draw axis titles
	const yAxisTitleX = padding + textSize / 2;
	const yAxisTitleY = (topExtent + bottomExtent) / 2;
	const xAxisTitleX = (leftExtent + rightExtent) / 2;
	const xAxisTitleY = canvas.height - yAxisTitleX;
	drawText(context, yTitle, yAxisTitleX, yAxisTitleY, textSize, 'center', font, colour, 270, '');
	drawText(context, xTitle, xAxisTitleX, xAxisTitleY, textSize, 'center', font, colour, 0, '');

	// Draw bins
	context.fillStyle = barColour;
	const binWidth = valueToX(binIncrement) - valueToX(0);
	const binLeft = valueToX(binMin);
	for (let i = 0; i < binValues.length; i++) {
		const value = binValues[i];
		const x = binLeft + binWidth * i;
		const y = valueToY(value);
		context.fillRect(x, y, binWidth, bottomExtent - y);
	}
}



/*==============================================================*/
/* Part 3 - Statistics
/*==============================================================*/
function updateStatisticsGeneral() {
	const lowest = CLIMATE_NET['lowest_point'];
	const highest = CLIMATE_NET['highest_point'];
	const farthestLand = CLIMATE_NET['farthest_land'];
	const farthestWater = CLIMATE_NET['farthest_water'];
	
	const [land, lower, upper] = percentLand(DATA_WATER);
	const formatAt = x => {
		const latitude = 90 - x[0];
		const longitude = 180 - x[1];
		const latitudeSymbol = latitude <= 0 ? 'N' : 'S';
		const longitudeSymbol = longitude <= 0 ? 'W' : 'E';
		return `${Math.abs(longitude)}˚ ${longitudeSymbol} ${Math.abs(latitude)}˚ ${latitudeSymbol}`;
	}

	const decimal = 2;
	rows = cssFindAll('#table-statistics-general tr');
	
	let values = (lowest[0] == -1) ? ['N/A', 'N/A'] : [formatAt(lowest), lowest[2].toFixed(decimal)];
	rows[1].children[1].innerText = values[0];
	rows[1].children[2].innerText = values[1];
	
	values = (highest[0] == -1) ? ['N/A', 'N/A'] : [formatAt(highest), highest[2].toFixed(decimal)];
	rows[2].children[1].innerText = values[0];
	rows[2].children[2].innerText = values[1];
	
	values = (farthestLand[0] == -1) ? ['N/A', 'N/A'] : [formatAt(farthestLand), farthestLand[2].toFixed(decimal)];
	rows[3].children[1].innerText = values[0];
	rows[3].children[2].innerText = values[1];
	
	values = (farthestWater[0] == -1) ? ['N/A', 'N/A'] : [`${(100 * lower).toFixed(decimal)} - ${(100 * upper).toFixed(decimal)}`, (100 * land).toFixed(decimal)];
	rows[5].children[1].innerText = values[0];
	rows[5].children[2].innerText = values[1];
	
	values = (farthestWater[0] == -1) ? ['N/A', 'N/A'] : [formatAt(farthestWater), farthestWater[2].toFixed(decimal)];
	rows[4].children[1].innerText = values[0];
	rows[4].children[2].innerText = values[1];
	
}
function areaFunction(latitude) {
	// let lat1 = (latitude + 0.5) * Math.PI / 180;
	// let lat2 = (latitude - 0.5) * Math.PI / 180;
	// return (Math.sin(lat1) - Math.sin(lat2)) / 0.01745307099674787;
	return Math.cos(Math.PI * latitude / 180);
}
function percentLand(water) {
	let landArea = 0;
	let totalArea = 0;
	let lowerArea = 0;
	let upperArea = 0;
	const isLand = (i, j) => !water[i][j];

	function getCertainty(i, j) {
		const neighbours = (i != 0) + (i != water.length - 1) + 2;
		const prev = (j - 1 + water[i].length) % water[i].length;
		const next = (j + 1) % water[i].length;
		let certainty = (isLand(i, prev) == isLand(i, j)) + (isLand(i, next) == isLand(i, j));
		if (i != 0)
			certainty += isLand(i - 1, j) == isLand(i, j);
		if (i != water.length - 1)
			certainty += isLand(i + 1, j) == isLand(i, j);
		return certainty / neighbours;
	}
	for (let i = 0; i < water.length; i++) {
		const latitude = Math.abs(i + 0.5 - 90);
		const area = areaFunction(latitude);
		totalArea += water[i].length * area;
		for (let j = 0; j < water[i].length; j++) {
			landArea += isLand(i, j) * area;
			const certainty = getCertainty(i, j);
			const uncertainty = 1 - certainty;
			if (isLand(i, j)) {
				upperArea += area;
				lowerArea += area * certainty;
			} else {
				upperArea += area * uncertainty;
			}
		}
	}
	return [landArea / totalArea, lowerArea / totalArea, upperArea / totalArea];
}
function getElevationHistogram(matrix, water, bounds, increment) {
	const [min, max] = bounds;
	let binMin = roundToFloat(min, 1 / increment);
	let binMax = roundToFloat(max, 1 / increment);
	if (min < binMin)
		binMin -= increment;
	if (max > binMax)
		binMax += increment;
	const numBins = (binMax - binMin) / increment + 1;

	const bins = [];
	for (let i = 0; i < numBins; i++) {
		bins.push(0);
	}
	let sum = 0;
	for (let i = 0; i < matrix.length; i++) {
		const latitude = Math.abs(i + 0.5 - 90);
		const area = areaFunction(latitude);
		for (let j = 0; j < matrix[i].length; j++) {
			if (water[i][j])
				continue;
			const prop = (matrix[i][j] - binMin) / (binMax - binMin);
			const bin = bound(Math.floor(prop * numBins), 0, numBins - 1);
			bins[bin] += area;
			sum += area;
		}
	}
	for (let i = 0; i < bins.length; i++) {
		bins[i] /= sum;
	}
	return [bins, binMin, binMax];
}
function getLatitudeHistogram(water) {
	const bins = [];
	for (let i = 0; i < 90; i++) {
		bins.push(0);
	}
	let sum = 0;
	for (let i = 0; i < water.length; i++) {
		const latitude = Math.abs(i + 0.5 - 90);
		const area = areaFunction(latitude);
		for (let j = 0; j < water[i].length; j++) {
			if (water[i][j])
				continue;
			bins[latitude - 0.5] += area;
			sum += area;
		}
	}
	for (let i = 0; i < bins.length; i++) {
		bins[i] /= sum;
	}
	return bins;
}


/*==============================================================*/
/* Part 3 - Download dropdown custom behaviour
/*==============================================================*/
function clickDownloadDropdown(event) {
	const element = event.srcElement;
	let parent = element;
	while (!parent.classList.contains('download-selector')) {
		parent = parent.parentElement;
	}

	// Open/close dropdown
	if (element.classList.contains('download-selector-active')) {
		const fileType = parent.id.substring(0, parent.id.lastIndexOf('-'));
		for (let options of cssFindAll(`#${fileType} .download-selector-options`)) {
			if (options == parent.children[1]) {
				const display = (options.style['display'] != 'flex') ? 'flex' : 'none';
				const maxHeight = window.innerHeight - element.getBoundingClientRect().bottom;
				cssSetElementToValues(options, ['display', display, 'max-height', `${maxHeight}px`]);
			} else {
				options.style.setProperty('display', 'none');
			}
		}

	// Set dropdown value
	} else if (element.classList.contains('download-selector-option')) {
		const value = element.innerText;
		const active = parent.children[0];
		active.innerText = value;

		if (parent.id == 'download-gif-item' || parent.id.includes('month'))
			return;
		const display = (value == 'Temperature' || value == 'Precipitation') ? 'inline-block' : 'none';
		const monthSelector = cssGetId(`${parent.id.substring(0, parent.id.lastIndexOf('-'))}-month`);
		monthSelector.parentElement.style.setProperty('display', display);
	}
}
function leaveDownloadDropdown(event) {
	const element = event.srcElement;
	const dropdowns = cssFindAll(`#${element.id} .download-selector-options`);
	for (let element of dropdowns) {
		element.style.setProperty('display', 'none');
	}
}


/*==============================================================*/
/* Part 3 - Saving, exporting, downloading
/*==============================================================*/
function enterSaveMenu() {
	cssSetId('save-overlay', 'opacity', 1);
	cssSetId('save-block', 'pointer-events', 'auto');
	const id = (CURR_PART == 2) ? 'export' : 'download';
	cssSetIdToValues(id, ['transform', 'scale(1)', 'opacity', '0.9']);
	cssGetId('copyable').value = matrixToString(DATA_FINAL);
}
function leaveSaveMenu() {
	cssSetId('save-overlay', 'opacity', 0);
	cssSetId('save-block', 'pointer-events', 'none');
	const id = (CURR_PART == 2) ? 'export' : 'download';
	cssSetIdToValues(id, ['transform', 'scale(0)', 'opacity', '0']);
}
function downloadData(event) {
	let element = event.srcElement;
	if (element.classList.contains('download-selector-active')
			|| element.classList.contains('download-selector-option')
			|| element.classList.contains('download-selector-options'))
		return;
	while (!element.classList.contains('file-format')) {
		element = element.parentElement;
	}
	if (element.id == 'download-gif')
		downloadGIF(element.id);
	else if (element.id == 'download-png')
		downloadPNG(element.id);
	else if (element.id == 'download-npy')
		downloadNPY(element.id);
	else if (element.id == 'download-txt')
		downloadTXT(element.id);
	else
		throw new Error(element.id);
}
function downloadGIF(id) {
	const item = cssFindFirst(`#${id}-item .download-selector-active`).innerText;
	const frameLength = getForm('frame-length');

	if (item == 'Temperature')
		saveGIF('climate-net-temperature.gif', 'temperature', frameLength);
	else if (item == 'Precipitation')
		saveGIF('climate-net-precipitation.gif', 'precipitation', frameLength);
	else
		throw new Error(item);
}
function downloadPNG(id) {
	const item = cssFindFirst(`#${id}-item .download-selector-active`).innerText;
	const month = cssFindFirst(`#${id}-month .download-selector-active`).innerText;
	const i = MONTHS['long'].indexOf(month);
	let longitude, latitude;
	if (LOCAL_PIXEL != null) {
		longitude = `${Math.abs(180 - LOCAL_PIXEL[0])}${LOCAL_PIXEL[0] <= 180 ? 'w' : 'e'}`;
		latitude = `${Math.abs(90 - LOCAL_PIXEL[1])}${LOCAL_PIXEL[1] <= 90 ? 'n' : 's'}`;
	}
	if (item == 'Temperature')
		postProcessPNG('temperature', i, () => savePNG(`climate-net-temperature-${i + 1}`, 'map-canvas'));
	else if (item == 'Precipitation')
		postProcessPNG('precipitation', i, () => savePNG(`climate-net-precipitation-${i + 1}`, 'map-canvas'));
	else if (item == 'Köppen-Geiger')
		postProcessPNG('koppen', null, () => savePNG('climate-net-koppen', 'map-canvas'));
	else if (item == 'Köppen-Trewartha')
		postProcessPNG('trewartha', null, () => savePNG('climate-net-trewartha', 'map-canvas'));
	else if (item == 'Local Climate')
		savePNG(`climate-net-local-${longitude}-${latitude}.png`, 'local-mini-canvas');
	else if (item == 'Area by Latitude')
		savePNG('climate-net-latitude-histogram.png', 'latitude-canvas');
	else if (item == 'Area by Elevation')
		savePNG('climate-net-elevation-histogram.png', 'elevation-canvas');
	else
		throw new Error(item);
}
function postProcessPNG(canvasDisplay, currMonth, f) {
	const temp = CURR_MONTH;
	CANVAS_DISPLAY = canvasDisplay;
	if (currMonth != null)
		CURR_MONTH = currMonth;
	drawResult();
	f();
	CURR_MONTH = temp;
	setCanvasSettings();
}
function downloadNPY(id) {
	const item = cssFindFirst(`#${id}-item .download-selector-active`).innerText;
	const month = cssFindFirst(`#${id}-month .download-selector-active`).innerText;
	const i = MONTHS['long'].indexOf(month);
	const monthShort = (i == -1) ? 'all' : MONTHS['short'][i];

	if (item == 'Temperature')
		saveNPY(`climate-net-temperature-${i + 1}.npy`, ...postProcessNPY(i, monthShort, 'temp'));
	else if (item == 'Precipitation')
		saveNPY(`climate-net-precipitation-${i + 1}.npy`, ...postProcessNPY(i, monthShort, 'prec'));
	else if (item == 'Köppen-Geiger')
		saveNPY(`climate-net-koppen.npy`, flattenMatrix(CLIMATE_NET['koppen']), [H, W]);
	else if (item == 'Köppen-Trewartha')
		saveNPY(`climate-net-trewartha.npy`, flattenMatrix(CLIMATE_NET['trewartha']), [H, W]);
	else
		throw new Error(item);
}
function postProcessNPY(i, monthShort, prefix) {
	if (i >= 0)
		return [flattenMatrix(CLIMATE_NET[`${prefix}_${monthShort}`]), [H, W]];
	const flattened = [];
	for (let month = 0; month < 12; month++) {
		const matrix = CLIMATE_NET[`${prefix}_${MONTHS['short'][month]}`];
		elementWiseIndexDo(matrix, (i, j) => {
			flattened.push(matrix[i][j]);
		});
	}
	return [flattened, [12, H, W]];
}
function downloadTXT(id) {
	const item = cssFindFirst(`#${id}-item .download-selector-active`).innerText;
	const month = cssFindFirst(`#${id}-month .download-selector-active`).innerText;
	const i = MONTHS['long'].indexOf(month);
	const monthShort = (i == -1) ? 'all' : MONTHS['short'][i];

	if (item == 'Temperature')
		saveTXT(`climate-net-temperature-${i + 1}.npy`, postProcessTXT(i, monthShort, 'temp'));
	else if (item == 'Precipitation')
		saveTXT(`climate-net-precipitation-${i + 1}.npy`, postProcessTXT(i, monthShort, 'prec'));
	else if (item == 'Köppen-Geiger')
		saveTXT(`climate-net-koppen.npy`, matrixToString(CLIMATE_NET['koppen']));
	else if (item == 'Köppen-Trewartha')
		saveTXT(`climate-net-trewartha.npy`, matrixToString(CLIMATE_NET['trewartha']));
	else
		throw new Error(item);
}
function postProcessTXT(i, monthShort, prefix) {
	if (i >= 0)
		return matrixToString(CLIMATE_NET[`${prefix}_${monthShort}`]);
	let str = '';
	for (let month = 0; month < 12; month++) {
		str += matrixToString(CLIMATE_NET[`${prefix}_${MONTHS['short'][month]}`]);
	}
	return str;
}
function exportData(event) {
	let element = event.srcElement;
	if (element.nodeName != 'SPAN')
		element = element.children[0];
	const value = element.innerText;
	if (value == 'PNG')
		savePNG('elevation-map.png', 'map-canvas');
	else if (value == 'NPY')
		saveNPY('elevation-map.npy', flattenMatrix(DATA_FINAL), [H, W]);
	else if (value == 'TXT')
		saveTXT('elevation-map.txt', cssGetId('copyable').value);
	else
		throw new Error(event.srcElement);
}


function saveGIF(path, canvasDisplay, frameLength) {
	const encoder = new GIFEncoder();
	encoder.setRepeat(0);
	encoder.setDelay(frameLength);
	encoder.setQuality(10);
	encoder.start();
	const temp = CURR_MONTH;
	CANVAS_DISPLAY = canvasDisplay;
	for (let i = 0; i < 12; i++) {
		CURR_MONTH = i;
		drawResult();
		encoder.addFrame(CONTEXT);
	}
	CURR_MONTH = temp;
	setCanvasSettings();
	encoder.finish();
	encoder.download(path);
}
function savePNG(path, canvasId) {
	ReImg.fromCanvas(cssGetId(canvasId)).downloadPng(path);
}
function saveNPY(path, flattened, shape) {
	const ndarray = {
		data: new Float32Array(flattened),
		shape: shape,
		fortran_order: false
	};
	const npyBuffer = npy.tobuffer(ndarray);
	const blob = new Blob([npyBuffer], {type: 'text/octet-stream'});
	const a = window.document.createElement('a');
	a.href = URL.createObjectURL(blob);
	a.download = path;
	document.body.appendChild(a);
	a.click();
	URL.revokeObjectURL(a.href);
	document.body.removeChild(a);
}
function saveTXT(path, text) {
	// https://stackoverflow.com/questions/40198508/prompt-user-to-download-output-as-txt-file
	const blob = new Blob([text], {type: 'text/plain'});
	const a = window.document.createElement('a');
	a.href = URL.createObjectURL(blob);
	a.download = path;
	document.body.appendChild(a);
	a.click();
	URL.revokeObjectURL(a.href);
	document.body.removeChild(a);
}

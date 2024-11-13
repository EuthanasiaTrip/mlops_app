<template>
  <div>
    <p style="color: red; font-size: x-large">
      Оценка вероятности неблагоприятного исхода
    </p>
    <div>
      <loading v-model:active="this.isShowLoading" :is-full-page="true" />
      <Modal
        v-model:visible="isModalVisible"
        title="Результат"
        :cancelButton="{ text: 'Отмена' }"
        :okButton="{ text: 'Принять' }"
      >
        <Line v-show="this.showChart" :data="this.chartData"> </Line>
        <div v-show="!this.showChart">
          <div
            style="
              display: flex;
              justify-content: flex-start;
              align-items: center;
              flex-direction: row;
            "
          >
            <p>
              Вероятность:
              {{ (covidNetPred * 100).toFixed(2) + "%" }}
            </p>
          </div>
        </div>
      </Modal>
      <Modal
        v-model:visible="showErrorModal"
        title="Ошибка"
        :cancelButton="{ text: 'Отмена' }"
        :okButton="{ text: 'Принять' }"
      >
        {{ errorMsg }}
      </Modal>
      <Modal
        v-model:visible="showInfoModal"
        title="Внимание"
        :cancelButton="{ text: 'Отмена' }"
        :okButton="{
          text: 'Предсказать',
          onclick: () => {
            this.callPredictionService();
          },
        }"
      >
        {{ infoMsg }}
      </Modal>
      <div
        style="
          display: flex;
          flex-direction: row;
          justify-content: space-between;
        "
      >
        <div
          style="
            display: flex;
            justify-content: flex-start;
            flex-direction: row;
            align-items: center;
          "
        >
          <button v-on:click="onClearInputClick" style="margin-right: 1%">
            Очистить поля
          </button>
          <button v-on:click="onSavePointClick" style="margin-right: 1%">
            Добавить точку
          </button>
          <button
            v-on:click="onPreviousPointClick"
            :disabled="currentPoint == 0"
            style="margin-right: 1%"
          >
            Предыдущая точка
          </button>
          <button
            v-on:click="onNextPointClick"
            :disabled="
              this.history.length == 0 ||
              this.currentPoint + 1 >= this.history.length
            "
            style="margin-right: 1%"
          >
            Следующая точка
          </button>
          <button v-on:click="onClearHistoryClick" style="margin-right: 1%">
            Удалить точки
          </button>
          <button v-on:click="onPredictClick" style="margin-right: 1%">
            Сделать предсказание
          </button>
          <p>
            {{ currentPointCaption }}
          </p>
        </div>
      </div>
    </div>
    <div class="input-container">
      <div class="input-layout-column">
        <div style="color: red">Общие показатели</div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Пол:</span>
          <select v-model="sex">
            <option disabled value="">Выберите пол</option>
            <option v-bind:value="false">Мужской</option>
            <option v-bind:value="true">Женский</option>
          </select>
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Возраст:</span>
          <input v-model="age" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Вес:</span>
          <input v-model="weight" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px">Курит</label>
          <input type="checkbox" id="checkbox-smoking" v-model="isSmoking" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Вакцинирован от COVID-19</label
          >
          <input type="checkbox" id="checkbox-covidvac" v-model="covidVac" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Вакцинирован от гриппа</label
          >
          <input type="checkbox" id="checkbox-fluvac" v-model="fluVac" />
        </div>
        <div style="color: red; padding-top: 15px">Клинические параметры</div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Температура:</span>
          <input v-model="maxTemp" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Максимальная степень ДН:</span>
          <select v-model="maxDNSeverityCategory">
            <option disabled value="">Выберите степень</option>
            <option>0</option>
            <option>1</option>
            <option>2</option>
            <option>3</option>
          </select>
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Частота дыхания:</span>
          <input v-model="maxBP" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Сатурация O2:</span>
          <input
            v-model="minSaturation"
            :placeholder="numberInputPlaceholder"
          />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px">Пневмония</label>
          <input type="checkbox" id="checkbox-pneumo" v-model="hasPneumo" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px"
            >Cтепень поражения лёгких по КТ :</span
          >
          <select v-model="maxKT">
            <option disabled value="">Выберите степень</option>
            <option>0</option>
            <option>1</option>
            <option>2</option>
            <option>3</option>
            <option>4</option>
          </select>
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Факт госпитализации в отделение реанимации</label
          >
          <input
            type="checkbox"
            id="checkbox-resus"
            v-model="wasInResuscitation"
          />
        </div>
        <div style="padding-top: 15px">
          <div v-show="wasInResuscitation">
            <label for="checkbox" style="padding-right: 15px">ИВЛ</label>
            <input type="checkbox" id="checkbox-ivl" v-model="wasOnIVL" />
          </div>
        </div>
        <div style="color: red">Хронические заболевания</div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Бронхиальная астма</label
          >
          <input type="checkbox" id="checkbox-asthma" v-model="hasAsthma" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Хроническая обструктивная болезнь легких</label
          >
          <input type="checkbox" id="checkbox-hobl" v-model="hasHOBL" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px">Диабет</label>
          <input type="checkbox" id="checkbox-diabetes" v-model="hasDiabetes" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px">Ожирение</label>
          <input type="checkbox" id="checkbox-obesity" v-model="hasObesity" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Артериальная гипертония</label
          >
          <input
            type="checkbox"
            id="checkbox-hypertonia"
            v-model="hasHypertonia"
          />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Ишемическая болезнь сердца</label
          >
          <input type="checkbox" id="checkbox-ibs" v-model="hasIBS" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Инфаркт миокарда</label
          >
          <input
            type="checkbox"
            id="checkbox-myocard"
            v-model="hasMyocardInfarct"
          />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Острое нарушение мозгового кровообращения</label
          >
          <input type="checkbox" id="checkbox-onmk" v-model="hasONMK" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px"
            >Почечная недостаточность</label
          >
          <input type="checkbox" id="checkbox-hpn" v-model="hasHPN" />
        </div>
        <div style="padding-top: 15px">
          <label for="checkbox" style="padding-right: 15px">Рак</label>
          <input type="checkbox" id="checkbox-cancer" v-model="hasCancer" />
        </div>
      </div>
      <div class="input-layout-column">
        <div style="color: red">Лабораторные показатели</div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Лимфоциты (абс):</span>
          <input v-model="minAbsLymph" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Лейкоциты (абс):</span>
          <input v-model="maxAbsLeic" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Гемоглобин:</span>
          <input
            v-model="minHemoglobin"
            :placeholder="numberInputPlaceholder"
          />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Тромбоциты (абс):</span>
          <input v-model="maxPlt" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">СОЭ:</span>
          <input v-model="maxESR" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Общий белок:</span>
          <input
            v-model="maxCommonProtein"
            :placeholder="numberInputPlaceholder"
          />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">С-реактивный белок:</span>
          <input v-model="maxCProtein" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Глюкоза:</span>
          <input v-model="maxGlucose" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Билирубин:</span>
          <input v-model="maxBilirubin" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Аланинаминотрансфераза:</span>
          <input v-model="maxALT" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Аспартатаминотрансфераза:</span>
          <input v-model="maxAST" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Мочевина:</span>
          <input v-model="maxUrea" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Креатинин:</span>
          <input
            v-model="maxCreatinine"
            :placeholder="numberInputPlaceholder"
          />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">МНО max:</span>
          <input v-model="maxMNO" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Ферритин:</span>
          <input v-model="maxFerritin" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Д-димер:</span>
          <input v-model="maxDDimer" :placeholder="numberInputPlaceholder" />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Протромбиновый индекс:</span>
          <input
            v-model="minProtrombIndex"
            :placeholder="numberInputPlaceholder"
          />
        </div>
        <div style="padding-top: 15px">
          <span style="padding-right: 15px">Фибриноген:</span>
          <input
            v-model="maxFibrinogen"
            :placeholder="numberInputPlaceholder"
          />
        </div>
      </div>
    </div>
  </div>
</template>

<script>
import { Modal } from "usemodal-vue3";
import {
  Chart as ChartJS,
  Title,
  Tooltip,
  Legend,
  LineElement,
  CategoryScale,
  LinearScale,
  PointElement,
} from "chart.js";
import { Line } from "vue-chartjs";

import Loading from "vue-loading-overlay";
import "vue-loading-overlay/dist/css/index.css";

ChartJS.register(
  CategoryScale,
  LinearScale,
  LineElement,
  Title,
  Tooltip,
  Legend,
  PointElement
);

export default {
  name: "MainWindow",
  components: { Modal, Line, Loading },
  data() {
    return {
      history: [],
      fieldsNames: [
        "sex",
        "age",
        "weight",
        "isSmoking",
        "minAbsLymph",
        "maxAbsLeic",
        "maxPlt",
        "maxESR",
        "maxCProtein",
        "maxFerritin",
        "maxDDimer",
        "maxUrea",
        "maxCommonProtein",
        "maxGlucose",
        "maxALT",
        "maxAST",
        "maxBilirubin",
        "maxMNO",
        "minProtrombIndex",
        "maxFibrinogen",
        "maxCreatinine",
        "minHemoglobin",
        "maxTemp",
        "minSaturation",
        "maxBP",
        "hasIBS",
        "hasMyocardInfarct",
        "hasONMK",
        "hasHypertonia",
        "hasHOBL",
        "hasDiabetes",
        "hasObesity",
        "hasHPN",
        "hasCancer",
        "hasHIV",
        "hasPneumo",
        "hasAsthma",
        "covidVac",
        "fluVac",
        "pneumococcusVac",
        "wasInResuscitation",
        "wasOnIVL",
        "maxDNSeverityCategory",
        "maxKT",
        "prediction",
      ],
      currentPoint: 0,
      currentPointCaption: "Точка 1",
      numberInputPlaceholder: "Введите число",
      sex: "",
      age: null,
      weight: null,
      isSmoking: false,
      maxDNSeverityCategory: "",
      hasIBS: false,
      hasMyocardInfarct: false,
      hasONMK: false,
      hasHypertonia: false,
      hasHOBL: false,
      hasDiabetes: false,
      hasObesity: false,
      hasHPN: false,
      hasCancer: false,
      hasHIV: false,
      hasPneumo: false,
      hasAsthma: false,
      covidVac: false,
      fluVac: false,
      pneumococcusVac: false,
      minAbsLymph: null,
      maxAbsLeic: null,
      maxPlt: null,
      maxESR: null,
      maxCProtein: null,
      maxFerritin: null,
      maxDDimer: null,
      maxUrea: null,
      maxCommonProtein: null,
      maxGlucose: null,
      maxALT: null,
      maxAST: null,
      maxBilirubin: null,
      maxMNO: null,
      minProtrombIndex: null,
      maxFibrinogen: null,
      maxCreatinine: null,
      minHemoglobin: null,
      maxTemp: null,
      minSaturation: null,
      maxBP: null,
      maxKT: "",
      wasOnIVL: false,
      wasInResuscitation: false,
      covidNetPred: 0,
      histgboostPred: 0,
      logisticregressionPred: 0,
      isModalVisible: false,
      chartData: {
        labels: ["0"],
        datasets: [
          {
            label: "Изменение вероятности летального исхода",
            backgroundColor: "#f87979",
            data: [],
          },
        ],
      },
      showErrorModal: false,
      showInfoModal: false,
      errorMsg: "",
      infoMsg: "",
      isShowLoading: false,
      showChart: false,
    };
  },
  methods: {
    onPredictClick() {
      this.covidNetPred = 0;
      this.histgboostPred = 0;
      this.logisticregressionPred = 0;
      const historyItem =
        this.currentPoint > 0 ? this.history.concat(this) : [this];
      if (this.checkHasNotNumValues(historyItem)) {
        this.errorMsg = "Введите только числовые значения";
        this.showErrorModal = true;
        return;
      }
      const hasEmptyData = this.checkHasEmptyFields(historyItem);
      console.log(hasEmptyData);
      this.callPredictionService();
    },

    callPredictionService() {
      const scope = this;
      this.isShowLoading = true;
      this.showInfoModal = false;
      fetch("http://localhost:8000/api/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(this.getCardData()),
      })
        .then(async function (response) {
          response = await response.json();
          if (response) {
            const predictions = response;
            let rmsValues = [];
            let rmsInput = [];
            predictions.forEach((item) => {
              rmsInput.push(item.pred);
            });
            for (let i = 0; i < rmsInput[0].length; i++) {
              let valuesToRms = [];
              rmsInput.forEach((values) => {
                valuesToRms.push(values[i]);
              });
              rmsValues.push(scope.calculateRMS(valuesToRms));
            }
            scope["covidNetPred"] = rmsValues;
            scope.isModalVisible = true;

            scope.updateChart(rmsValues);
          } else if (!response) {
            scope.showErrorModal = true;
            scope.errorMsg = "Произошла ошибка при запросе к модели:";
            scope.errorMsg += "\n" + response.err;
          }
          scope.isShowLoading = false;
        })
        .catch(async function (error) {
          scope.isShowLoading = false;
          scope.showErrorModal = true;
          scope.errorMsg = "Произошла ошибка при запросе к модели:";
          scope.errorMsg += "\n" + error;
        });
    },

    checkHasEmptyFields(points) {
      const numericFields = [
        "sex",
        "age",
        "weight",
        "minAbsLymph",
        "maxAbsLeic",
        "maxPlt",
        "maxESR",
        "maxCProtein",
        "maxFerritin",
        "maxDDimer",
        "maxUrea",
        "maxCommonProtein",
        "maxGlucose",
        "maxALT",
        "maxAST",
        "maxBilirubin",
        "maxMNO",
        "minProtrombIndex",
        "maxFibrinogen",
        "maxCreatinine",
        "minHemoglobin",
        "maxTemp",
        "minSaturation",
        "maxBP",
        "maxDNSeverityCategory",
        "maxKT",
      ];
      let hasEmpty = false;
      const columnsToGet = this.fieldsNames.filter((name) =>
        numericFields.includes(name)
      );
      points.forEach((point) => {
        columnsToGet.forEach((column) => {
          if (!hasEmpty) {
            hasEmpty = point[column] === "" || point[column] === null;
          }
        });
      });
      return hasEmpty;
    },

    checkHasNotNumValues(points) {
      const numericFields = [
        "age",
        "weight",
        "minAbsLymph",
        "maxAbsLeic",
        "maxPlt",
        "maxESR",
        "maxCProtein",
        "maxFerritin",
        "maxDDimer",
        "maxUrea",
        "maxCommonProtein",
        "maxGlucose",
        "maxALT",
        "maxAST",
        "maxBilirubin",
        "maxMNO",
        "minProtrombIndex",
        "maxFibrinogen",
        "maxCreatinine",
        "minHemoglobin",
        "maxTemp",
        "minSaturation",
        "maxBP",
      ];
      let hasNotValid = false;
      const columnsToGet = this.fieldsNames.filter((name) =>
        numericFields.includes(name)
      );
      points.forEach((point) => {
        columnsToGet.forEach((column) => {
          if (!hasNotValid) {
            hasNotValid = isNaN(point[column]);
          }
        });
      });
      return hasNotValid;
    },

    getCardData() {
      if (this.currentPoint === this.history.length) {
        this.writeHistory();
      } else {
        this.history[this.currentPoint] = this.saveHistory();
      }
      const columnsToGet = this.fieldsNames.filter(
        (name) =>
          !["isPregnant", "prediction", "minMNO", "isEmployed"].includes(name)
      );
      let dataArray = [];
      let hasEmptyData = false;
      this.history.forEach((item) => {
        let dataObject = {};
        columnsToGet.forEach((name) => {
          let value = this.castValue(item[name]);
          let upperCaseName = name.charAt(0).toUpperCase() + name.slice(1);
          dataObject[upperCaseName] = value;
        });
        if (!hasEmptyData) {
          hasEmptyData = this.checkHasEmptyFields([item]);
        }
        dataArray.push(dataObject);
      });
      return { hasEmptyData: hasEmptyData, data: dataArray };
    },

    updateChart(predictions) {
      const dataLen = predictions.length;
      if (dataLen > 1) {
        this.showChart = true;
        const dataSets = [
          {
            label: "Вероятность, %",
            borderColor: this.getChartColor("covidNet"),
            backgroundColor: this.getChartColor("covidNet"),
            data: predictions.map((i) => (i * 100).toFixed(2)),
          },
        ];

        this.chartData = {
          labels: [...Array(dataLen).keys()],
          datasets: dataSets,
        };
      } else {
        this.showChart = false;
      }
    },

    getChartColor(modelName) {
      let color = "#000000";
      switch (modelName) {
        case "covidNet":
          color = "#0080FF";
          break;
        case "histgboost":
          color = "#FF007F";
          break;
        case "logisticregression":
          color = "#CCCC00";
          break;
      }
      return color;
    },

    getDataFormData() {
      const columnsToGet = this.fieldsNames.filter(
        (name) =>
          !["isPregnant", "prediction", "minMNO", "isEmployed"].includes(name)
      );
      let data = {};
      columnsToGet.forEach((name) => {
        let value = this.castValue(this[name]);
        let upperCaseName = name.charAt(0).toUpperCase() + name.slice(1);
        data[upperCaseName] = value;
      });
      return data;
    },

    castValue(value) {
      let val = value;
      if (typeof value === "string") {
        val = value.replace(/,/g, ".");
      }
      return Number(val);
    },

    onSavePointClick() {
      if (
        this.history.length != 0 &&
        this.currentPoint != this.history.length
      ) {
        this.history[this.currentPoint] = this.saveHistory();
      } else {
        this.writeHistory();
      }
      this.currentPoint = this.history.length;
      this.prediction = 0;
    },

    onClearInputClick() {
      this.fieldsNames.forEach((item) => {
        this[item] = "";
      });
      this.prediction = 0;
    },

    onPreviousPointClick() {
      if (this.currentPoint === this.history.length) {
        this.writeHistory();
      } else {
        this.history[this.currentPoint] = this.saveHistory();
      }
      this.currentPoint--;
      this.loadHistory();
    },

    onNextPointClick() {
      this.history[this.currentPoint] = this.saveHistory();
      this.currentPoint++;
      this.loadHistory();
    },

    onClearHistoryClick() {
      this.history = [];
      this.currentPoint = 0;
      this.currentPointCaption = "Точка 1";
    },

    saveHistory() {
      let historyObj = {};
      this.fieldsNames.forEach((fieldName) => {
        historyObj[fieldName] = this[fieldName];
      });
      return historyObj;
    },

    writeHistory() {
      const historyObj = this.saveHistory();
      this.history.push(historyObj);
    },

    loadHistory() {
      const historyObj = this.history[this.currentPoint];
      this.fieldsNames.forEach((fieldName) => {
        this[fieldName] = historyObj[fieldName];
      });
    },

    calculateRMS(values) {
      const sqrs = values.map((i) => Math.pow(i, 2));
      const sum = sqrs.reduce((partialSum, a) => partialSum + a, 0);
      return Math.sqrt(sum);
    },
  },
  watch: {
    currentPoint() {
      const historyCount = this.history.length;
      if (historyCount != 0) {
        this.currentPointCaption =
          this.currentPoint >= historyCount
            ? `Точка ${this.currentPoint + 1}`
            : `Точка ${this.currentPoint + 1} из ${historyCount}`;
      } else {
        this.currentPointCaption = "Точка 1";
      }
    },

    getChartData() {
      const pointsNumber = this.chartData.length;
      return {
        labels: [...Array(pointsNumber).keys()],
        datasets: [
          {
            label: "prediction",
            data: this.chartData,
          },
        ],
      };
    },
  },
};
</script>

<style lang="css">
.input-container {
  display: flex;
  flex-direction: row;
  justify-content: space-evenly;
}
.input-layout-column {
  display: flex;
  flex-direction: column;
  height: 100%;
  padding-top: 2%;
  align-items: flex-end;
}
</style>

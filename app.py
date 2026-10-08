
            # Create input features

            features = np.array(

                [[
                    radius,
                    perimeter,
                    area
                ]]

            )


            # ------------------------------------------------
            # PREDICT
            # ------------------------------------------------

            result = model.predict(
                features
            )[0]


            # 0 = Malignant
            # 1 = Benign

            if int(result) == 0:

                prediction = "Malignant"

            else:

                prediction = "Benign"


            # ------------------------------------------------
            # MODEL PROBABILITY
            # ------------------------------------------------

            if hasattr(
                model,
                "predict_proba"
            ):

                probabilities = model.predict_proba(
                    features
                )[0]


                probability = round(

                    max(probabilities) * 100,

                    2

                )


        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            prediction = "Prediction Error"


    # ========================================================
    # SEND TO PREDICTION PAGE
    #
    # IMPORTANT:
    # Graph + prediction are shown ONLY here.
    # ========================================================

    return render_template(

        "predict.html",

        prediction=prediction,

        probability=probability,

        radius=radius,

        perimeter=perimeter,

        area=area

    )


# ============================================================
# RUN FLASK APPLICATION
# ============================================================

if __name__ == "__main__":

    # Local development only.
    # Render uses Gunicorn with: gunicorn app:app
    port = int(os.environ.get("PORT", 5000))

    app.run(
        debug=False,
        host="0.0.0.0",
        port=port
    )

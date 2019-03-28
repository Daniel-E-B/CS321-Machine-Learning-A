#pragma once

#include <vector>

#include <SFML/Graphics.hpp>

#include "Basket.hpp"
#include "Drawable.hpp"
#include "Food.hpp"

class Game : public Drawable {
   private:
    std::vector<Food> foods;
    Basket *basket;
    int currentFood = 0;

   public:
    void draw(sf::RenderWindow &);
    Game(int, sf::RenderWindow &);  // num foods
    bool tick();                    // returns true if generation is complete
    void mutate();                  // probably have parameters
    void setFood(int);              // food # in array TODO: rename food to fruit
    std::vector<double> getFitness();
    double getAvgFitness();
};
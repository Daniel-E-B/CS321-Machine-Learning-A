#pragma once

#include <SFML/Graphics.hpp>

#include "tiny-dnn/tiny_dnn/tiny_dnn.h"

class Basket {  // TODO: rename to Catcher or Chaser or something like that
   private:
    sf::RectangleShape shape;
    tiny_dnn::network<tiny_dnn::sequential> net;
    const sf::Color COLOR = sf::Color(127, 127, 0);
    const double SIZE = 30;  // width
    const int MAX_SPEED_COMPONENT = 5;
    std::vector<float> recurrentOuts;

   public:
    Basket();
    void draw(sf::RenderWindow &);
    void move(sf::Vector2f const &);
    sf::Vector2f getPos();
    std::string getBrain();
    void setBrain(std::string &);
    void setPos(sf::Vector2f &);
};
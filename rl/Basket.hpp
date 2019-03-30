#pragma once

#include <SFML/Graphics.hpp>

#include "GameObject.hpp"

class Basket : public GameObject { // TODO: rename to Catcher or Chaser or something like that
   private:
    sf::RectangleShape shape;
    const sf::Color COLOR = sf::Color(127, 127, 0);
    const double SIZE = 30; // width

   public:
    Basket(sf::Vector2f &);
    void draw(sf::RenderWindow &);
    void move(sf::Vector2f &);
};
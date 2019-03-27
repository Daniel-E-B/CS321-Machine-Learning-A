#include "Basket.hpp"

Basket::Basket(sf::Vector2f &pos) {
    shape = sf::RectangleShape(*(new sf::Vector2f(SIZE, SIZE / 10)));
    shape.setOrigin(SIZE / 2, SIZE/10);
    shape.setFillColor(COLOR);
    shape.setPosition(pos);
}

void Basket::draw(sf::RenderWindow &window) {
    window.draw(shape);
}

void Basket::move(sf::Vector2f &offset) {
    shape.move(offset);
}
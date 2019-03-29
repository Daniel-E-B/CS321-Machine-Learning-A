#include "Basket.hpp"

Basket::Basket(sf::Vector2f &pos) {
    shape = sf::RectangleShape(*(new sf::Vector2f(SIZE, SIZE)));
    shape.setOrigin(SIZE / 2, SIZE / 2);
    shape.setFillColor(COLOR);
    shape.setPosition(pos);
}

void Basket::draw(sf::RenderWindow &window) {
    window.draw(shape);
}

void Basket::move(sf::Vector2f &offset) {
    shape.move(offset);
}
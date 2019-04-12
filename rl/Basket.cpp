#include "Basket.hpp"
#include "tiny-dnn/tiny_dnn/tiny_dnn.h"

Basket::Basket() {
    shape = sf::RectangleShape(*(new sf::Vector2f(SIZE, SIZE)));
    shape.setOrigin(SIZE / 2, SIZE / 2);
    shape.setFillColor(COLOR);
    recurrentOuts.resize(2);
    recurrentOuts = {0.0, 0.0};

    net << tiny_dnn::layers::fc(6, 5) << tiny_dnn::activation::leaky_relu()
        << tiny_dnn::layers::fc(5, 5) << tiny_dnn::activation::leaky_relu()
        << tiny_dnn::layers::fc(5, 4) << tiny_dnn::activation::tanh();
}

void Basket::draw(sf::RenderWindow &window) {
    window.draw(shape);
}

void Basket::move(sf::Vector2f const &targetPos) {                                                                                                     // TODO: make more things const when they don't need to change
    tiny_dnn::vec_t pred = net.predict({shape.getPosition().x, shape.getPosition().y, targetPos.x, targetPos.y, recurrentOuts[0], recurrentOuts[1]});  // current pos, targetpos, recurrent
    shape.move(*(new sf::Vector2f(pred[0], pred[1])));
    recurrentOuts = {pred[2], pred[3]};
}

sf::Vector2f Basket::getPos() {
    return shape.getPosition();
}

std::string Basket::getBrain() {
    return net.to_json();
}

void Basket::setBrain(std::string &brain) {
    net.from_json(brain);
}

void Basket::setPos(sf::Vector2f &pos) {
    shape.setPosition(pos);
}
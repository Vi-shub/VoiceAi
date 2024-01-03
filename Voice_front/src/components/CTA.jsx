import styles from "../style";
import Button from "./Button";

const CTA = () => (
  <section
    className={`${styles.flexCenter} ${styles.marginY} ${styles.padding} sm:flex-row flex-col bg-black-gradient-2 rounded-[20px] box-shadow`}
  >
    <div className="flex-1 flex flex-col">
      <h2 className={styles.heading2}>Let’s try our Voice assistant now!</h2>
      <p className={`${styles.paragraph} max-w-[470px] mt-5`}>
        Empower your meetings with our AI, offering everything you need to
        enhance collaboration and communication.
      </p>
    </div>

    <div className={`${styles.flexCenter} sm:ml-10 ml-0 sm:mt-0 mt-10`}>
      <a href="/">
        <Button />
        </a>
    </div>
  </section>
);

export default CTA;
